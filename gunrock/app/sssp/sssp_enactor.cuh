// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

// Multi-stream Specific - Include
//#include <gunrock/app/enactor_types.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <unistd.h>
#include <gunrock/app/frontier.cuh>
#include <moderngpu.cuh>

// Graph Coloring Specific - Include
#include <gunrock/app/color/color_enactor.cuh>
#include <gunrock/app/color/color_test.cuh>
#include <string>
#include <iostream>

namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of SSSP iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SSSPIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push |
                                             (((EnactorT::Problem::FLAG &
                                                Mark_Predecessors) != 0)
                                                  ? Update_Predecessors
                                                  : 0x0)> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push |
                                          (((EnactorT::Problem::FLAG &
                                             Mark_Predecessors) != 0)
                                               ? Update_Predecessors
                                               : 0x0)>
      BaseIterationLoop;

  SSSPIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of sssp, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // DEBUG
    // printf("Executing Core ...\n");
    // Data sssp that works on
    // auto &contexts = this->enactor->contexts;
    // auto &frontiers = this->enactor->frontiers;
    // auto &streams = this->enactor->streams;
    auto &multistream_enactor_slices =
        this->enactor->multistream_enactor_slices;
    int num_stream = this->enactor->problem->num_stream;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &labels = data_slice.labels;

    //Multi-stream needs MemcpyAsync
    auto &stream_distances = this->enactor->stream_distances;
    //auto &stream_preds = this->enactor->stream_preds;
    auto &preds = data_slice.preds;
    auto &distances = data_slice.distances;

    // auto         &row_offsets        =   graph.CsrT::row_offsets;
    auto &weights = graph.CsrT::edge_values;
    auto &original_vertex = graph.GpT::original_vertex;
    auto &frontier0 = enactor_slice.frontier;
    auto &stream0 = enactor_slice.stream;
    auto &oprtr_parameters0 = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    // auto         &stream             =   enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;
    auto &color_in = data_slice.color_in;
    auto &source = data_slice.src;
/*==============================================================================
Multi-stream SSSP with 1 time coloring during Reset()
==============================================================================*/
    if (color_in) {

      auto advance_op =
          [distances, weights, original_vertex, preds, iteration,source] __host__ __device__(
              const VertexT &src, VertexT &dest, const SizeT &edge_id,
              const VertexT &input_item, const SizeT &input_pos,
              SizeT &output_pos) -> bool {
        ValueT src_distance = Load<cub::LOAD_CG>(distances + src);
        ValueT edge_weight = Load<cub::LOAD_CS>(weights + edge_id);
        ValueT new_distance = src_distance + edge_weight;

        if (iteration == 0 && src != source)
          return false;
        // Check if the destination node has been claimed as someone's child
        ValueT old_distance = atomicMin(distances + dest, new_distance);

        // printf("max num = %d \n", util::PreDefinedValues<ValueT>::MaxValue);

        if (new_distance < old_distance) {
         // if (!preds.isEmpty()) {
         //   VertexT pred = src;
         //   if (!original_vertex.isEmpty()) pred = original_vertex[src];
         //   Store(preds + dest, pred);
         // }
          return true;
        }
        return false;
      };

      // The filter operation
      auto filter_op = [labels, iteration] __host__ __device__(
                           const VertexT &src, VertexT &dest,
                           const SizeT &edge_id, const VertexT &input_item,
                           const SizeT &input_pos, SizeT &output_pos) -> bool {

        //printf("At iteration %d, src = %d, dest = %d, labels[dest] = %d\n", iteration, src, dest, labels[dest]);

        if (!util::isValid(dest)) return false;
        if (labels[dest] == iteration) return false;
        labels[dest] = iteration;
        return true;
      };
/*==============================================================================
Multi-stream Call Advance, distances and preds are different for each stream
==============================================================================*/
      for (int i = 0; i < num_stream; i++) {

        auto &oprtr_parameters = multistream_enactor_slices[i].oprtr_parameters;
        auto &frontier = multistream_enactor_slices[i].frontier;
        auto &stream = multistream_enactor_slices[i].stream;
        oprtr_parameters.label = iteration + 1;

        // Call the advance operator, using the advance operation
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
            advance_op, filter_op));
      }

      // for (int i = 0; i < num_stream; i++) {
      //   auto &stream = multistream_enactor_slices[i].stream;
      //   GUARD_CU2(cudaStreamSynchronize(stream),
      //             "cudaStreamSynchronize failed");
      // }

      auto & oprtr_parameters = multistream_enactor_slices[0].oprtr_parameters;
      if (oprtr_parameters.advance_mode != "LB_CULL" &&
          oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
/*==============================================================================
Multi-stream Call Filter
==============================================================================*/
        // printf("DEBUG: Launch Filer Op \n");
	for (int i = 0; i < num_stream; i++) {

          auto &oprtr_parameters =
              multistream_enactor_slices[i].oprtr_parameters;
          auto &frontier = multistream_enactor_slices[i].frontier;
          frontier.queue_reset = false;
          GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
              graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
              oprtr_parameters, filter_op));
        }
        // for (int i = 0; i < num_stream; i++) {
        //   auto &stream = multistream_enactor_slices[i].stream;
        //   GUARD_CU2(cudaStreamSynchronize(stream),
        //             "cudaStreamSynchronize failed");
        // }
      }

/*==============================================================================
Multi-stream Call queue update
==============================================================================*/
      // printf("DEBUG: Update Queue Length \n");
      for (int i = 0; i < num_stream; i++) {
        // auto &frontier = frontiers[i];
        auto &frontier = multistream_enactor_slices[i].frontier;
        auto &stream = multistream_enactor_slices[i].stream;
        // Get back the resulted frontier length
        GUARD_CU(frontier.work_progress.GetQueueLength(
            frontier.queue_index, frontier.queue_length, false, stream,
            true));
      }

    }

/*==============================================================================
 Multistreaming SSSP with dynamic coloring based on queue_length
==============================================================================*/
    else {

      // If it is first iteration, take frontier from normal sssp
      auto &frontier = frontier0;
      auto &oprtr_parameters = oprtr_parameters0;
      auto &stream  = stream0;
/*==============================================================================
 Perform normal sssp if the queue_length is too small TODO: make threshold a parameter
==============================================================================*/
      if (false)	{
    	  // The advance operation
    	  auto advance_op =
    	      [distances, weights, original_vertex, preds, iteration,source] __host__ __device__(
    	          const VertexT &src, VertexT &dest, const SizeT &edge_id,
    	          const VertexT &input_item, const SizeT &input_pos,
    	          SizeT &output_pos) -> bool {
    	    ValueT src_distance = Load<cub::LOAD_CG>(distances + src);
    	    ValueT edge_weight = Load<cub::LOAD_CS>(weights + edge_id);
    	    ValueT new_distance = src_distance + edge_weight;

    	    if (iteration == 0 && src != source)
    	      return false;
    	    // Check if the destination node has been claimed as someone's child
    	    ValueT old_distance = atomicMin(distances + dest, new_distance);

    	    // printf("max num = %d \n", util::PreDefinedValues<ValueT>::MaxValue);

    	    if (new_distance < old_distance) {
    	     // if (!preds.isEmpty()) {
    	     //   VertexT pred = src;
    	     //   if (!original_vertex.isEmpty()) pred = original_vertex[src];
    	     //   Store(preds + dest, pred);
    	     // }
    	      return true;
    	    }
    	    return false;
    	  };

    	  // The filter operation
    	  auto filter_op = [labels, iteration] __host__ __device__(
    	                       const VertexT &src, VertexT &dest,
    	                       const SizeT &edge_id, const VertexT &input_item,
    	                       const SizeT &input_pos, SizeT &output_pos) -> bool {


    	    if (!util::isValid(dest)) return false;
    	    if (labels[dest] == iteration) return false;
    	    labels[dest] = iteration;
    	    return true;
    	  };

    	  oprtr_parameters.label = iteration + 1;
    	  // Call the advance operator, using the advance operation
    	  GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
    	      graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
    	      advance_op, filter_op));

    	  if (oprtr_parameters.advance_mode != "LB_CULL" &&
    	      oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
    	    frontier.queue_reset = false;
    	    // Call the filter operator, using the filter operation
    	    GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
    	        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
    	        filter_op));
    	  }

    	  // Get back the resulted frontier length
    	  GUARD_CU(frontier.work_progress.GetQueueLength(
    	      frontier.queue_index, frontier.queue_length, false,
    	      oprtr_parameters.stream, true));
    	} //end if frontier.queue_length >= threshold

/*==============================================================================
 Partition the frontier into multiple stream if frontier is too large
==============================================================================*/
	else {
	
	    std::cout << "DEBUG : start multi stream on iteration " << iteration << std::endl;

	    // create color array for frontier
	    int color_iteration;
	    util::Array1D<SizeT, VertexT> colors;
	    colors.SetName("colors");
	    colors.Allocate(graph.nodes, util::DEVICE | util::HOST);

	    // create rand array for frontier
	    util::Array1D<SizeT, float> rand;
	    rand.SetName("rand");
	    rand.Allocate(graph.nodes, util::DEVICE);
	    curandGenerator_t gen;
	    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

	    //TODO make seed a parameter (seed = 0 now)
	    curandSetPseudoRandomGeneratorSeed(gen, 0);
	    curandGenerateUniform(gen, rand.GetPointer(util::DEVICE), graph.nodes);

	    // create omit array
	    util::Array1D<SizeT, bool>  omit;
	    omit.SetName("omit");
	    omit.Allocate(graph.nodes, util::DEVICE | util::HOST);	    

	    // initialized omit (use stream 1) - this option only make sense with at least 2 streams
            GUARD_CU(omit.ForEach(
            [] __host__ __device__(bool & x){
                x = true;
            }, graph.nodes, util::DEVICE, stream));

	    // initialzed colors (use stream 2) - this option only make sense with at least 2 streams
	    GUARD_CU(colors.ForEach(
            [] __host__ __device__(VertexT & x) {
                x = util::PreDefinedValues<VertexT>::InvalidValue;
            }, graph.nodes, util::DEVICE, stream));

	    // define omit lambda (use stream 1) 
	    auto omit_neighbor = [omit] __host__ __device__
	    (VertexT * v_q, const SizeT &pos){
		omit[v_q[pos]] = false;		
	    };

	    // populate omit array (use stream 1)
	    GUARD_CU(frontier.V_Q()->ForAll(omit_neighbor, frontier.queue_length,
					    util::DEVICE, stream));

	    // define fast coloring lambda
            auto fast_coloring = [color_iteration, colors, rand, graph, omit] __host__ __device__
            (VertexT * v_q, const SizeT &pos){
		// pos is in order,  v_q[pos] is actual graph index
		VertexT global_idx = v_q[pos];
		if (util::isValid(colors[pos])) return;
		SizeT start_edge = graph.CsrT::GetNeighborListOffset(global_idx);
		SizeT num_neighbors = graph.CsrT::GetNeighborListLength(global_idx);
		bool colormax = true;
		//bool colormin = true;
		//int color = color_iteration * 2;
		for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
			VertexT u = graph.CsrT::GetEdgeDest(e);

			// if this neighbor is not in frontier, skip it
			if (omit[u]) continue;

			if ((util::isValid(colors[u])) && (colors[u] != color_iteration)  
        		    || (global_idx == u)) continue;                                                   
		        if (rand[global_idx] <= rand[u]) colormax = false;                       
                                                
		        // if (rand[global_idx] >= rand[u]) colormin = false;                     
                                                              
		}

		if (colormax) colors[global_idx] = color_iteration;
		// if (colormin && !colormax) colors[global_idx] = color + 2;
            };
	
            // populate colors array
	    for (color_iteration = 0 ; color_iteration < num_stream; color_iteration++) {
		GUARD_CU(frontier.V_Q()->ForAll(fast_coloring, frontier.queue_length, util::DEVICE, stream));
	    }

	    // populate multistream frontiers
	    for (int color = 0; color < num_stream; color += 1) {
		auto & output_frontier = multistream_enactor_slices[color].frontier;
		output_frontier.Reset();
		output_frontier.Allocate(frontier.queue_length, graph.edges, this->enactor->queue_factors);
		output_frontier.queue_length = frontier.queue_length;
		auto & output_V_Q  = *(output_frontier.V_Q());
		//GUARD_CU(output_V_Q.ForEach([] __host__ __device__(VertexT & x){
		//	x = util::PreDefinedValues<VertexT>::InvalidValue;
		//}, frontier.queue_length, util::DEVICE, multistream_enactor_slices[color].stream));

	        auto partition = [omit, color, colors, output_V_Q, num_stream] __host__ __device__
	        (VertexT * v_q, const SizeT &pos){
		    VertexT global_idx = v_q[pos];
		    //if (omit[global_idx]) return;
		    if (color == colors[global_idx] && color < num_stream - 1)
		 	output_V_Q[pos] = global_idx; 
		    else if (!util::isValid(colors[global_idx]) && color == num_stream - 1)
			output_V_Q[pos] = global_idx;
		    else
			output_V_Q[pos] = util::PreDefinedValues<VertexT>::InvalidValue;
				
	        };
		frontier.V_Q()->ForAll(partition, frontier.queue_length, util::DEVICE, multistream_enactor_slices[color].stream);

            }

	   // colors array is not needed at this point 
	   GUARD_CU(colors.Release(util::DEVICE | util::HOST));
	   GUARD_CU(rand.Release(util::DEVICE | util::HOST));
           GUARD_CU(omit.Release(util::DEVICE | util::HOST));
/*==============================================================================
SSSP Routine with Dynamic MultiStream
==============================================================================*/
	   
	    auto filter_op = [labels, iteration] __host__ __device__(
                                const VertexT & src, VertexT &dest,
                                const SizeT &edge_id, const VertexT &input_item,
                                const SizeT &input_pos, SizeT output_pos) -> bool {
	             printf("Inside filter op\n");
                     if (!util::isValid(dest)) return false;
	             printf("labels[%d] = %d \n", dest, labels[dest]);
                     if (labels[dest] == iteration) return false;
                     labels[dest] = iteration;
                     return true;
                 };
 
	    // Duplicate multiple distances array, one for each stream
	    distances.Move(util::DEVICE, util::HOST, graph.nodes);
	    for (int i = 0; i < num_stream; i++) {
		stream_distances[i].SetPointer(distances.GetPointer(util::HOST),
	        graph.nodes, util::HOST);
		stream_distances[i].Move(util::HOST, util::DEVICE, graph.nodes, 0,
		multistream_enactor_slices[i].stream);
	    }

	    std::cout << "Before Advance" << std::endl;
            std::cout << "queue_length = " << frontier.queue_length << std::endl;

	    // Multistream advance for loop
	    SizeT total_length = (SizeT) 0;
	    SizeT * lengths = new SizeT[num_stream + 1];
	    lengths[0] = (SizeT) 0;
	    for (int i = 0; i < num_stream; i++) {
		
		 auto & distances_ = stream_distances[i];
           	 auto advance_op = 
	   	[distances_, weights, original_vertex, preds, iteration, source] __host__ __device__(
	   	     	const VertexT &src, VertexT &dest, const SizeT &edge_id,
	   	     	const VertexT &input_item, const SizeT &input_pos,
	   	     	SizeT &output_pos) -> bool {
	   	     if (!util::isValid(src))
	   	     	return false;
	   	     
	   	     ValueT src_distance = Load<cub::LOAD_CG>(distances_ + src);
           	     ValueT edge_weight = Load<cub::LOAD_CS>(weights + edge_id);
           	     ValueT new_distance = src_distance + edge_weight;

	   	     ValueT old_distance = atomicMin(distances_ + dest, new_distance);

	   	     if (new_distance < old_distance) return true;
	   	     else return false;
	   	 };

	         // Launching advance op - Timed for performance
		 auto &oprtr_parameters_ = multistream_enactor_slices[i].oprtr_parameters;
                 auto &frontier_ = multistream_enactor_slices[i].frontier;
                 auto &stream_ = multistream_enactor_slices[i].stream;
        	 oprtr_parameters.label = iteration + 1;

        	 // Call the advance operator, using the advance operation
        	 GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            	 graph.csr(), frontier_.V_Q(), frontier_.Next_V_Q(), oprtr_parameters_,
            	 advance_op, filter_op));

		// Update the frontier length for reduction op later on ...
		GUARD_CU(frontier_.work_progress.GetQueueLength(
			 frontier_.queue_index, frontier_.queue_length, false, stream_, true));
		total_length += frontier_.queue_length;
		lengths[i + 1] = frontier_.queue_length;
		std::cout << "lengths["<< i+1 << "]"<< frontier_.queue_length << std::endl;
	   }

	std::cout << "Total length is: " << total_length << std::endl;

	std::cout << "Before Reduction" << std::endl;
        std::cout << "queue_length = " << frontier.queue_length << std::endl;
	for (int i = 0; i < num_stream; i++) {
		auto & distances_ = stream_distances[i];
		auto & frontier_ = multistream_enactor_slices[i].frontier;
		auto & input_V_Q = (*frontier_.V_Q());
		auto start = lengths[i];
		auto size  = lengths[i+1];
		auto reduce_op =
		[distances_, distances, start, input_V_Q, size] 
		__host__ __device__ (VertexT * v_q, const SizeT &pos) {
			// Merge distance array
			if (distances_[pos] < distances[pos])
				distances[pos] = distances_[pos];
			// Merge frontier array
			if (start + pos < start + size) {
				v_q[start + pos] = input_V_Q[pos];
			}
		};

		// Launch reduce op
		GUARD_CU(frontier.Next_V_Q()->ForAll(reduce_op, graph.nodes, util::DEVICE, stream));
	}
	
	//cudaStreamSynchronize(stream);	
	//frontier.queue_index += 1;

	//GUARD_CU(frontier.work_progress.GetQueueLength(
        //    frontier.queue_index, frontier.queue_length, true, stream,
        //    true));
        

	std::cout << "Before Filter" << std::endl;
        std::cout << "queue_length = " << frontier.queue_length << std::endl;

	if (oprtr_parameters.advance_mode != "LB_CULL" &&
	    oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
		frontier.queue_reset = false;

		// Launching filter op - Timed for performance
    	        GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
    	            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
    	            filter_op));
	}

	// auto printBug =  []
        //__host__ __device__ (VertexT * v_q, const SizeT &pos) {
        //        printf("v_q[%d] = %d \n", pos, v_q[pos]);
        //};

	//std::cout << "Before work progress " << std::endl;;
	//GUARD_CU(frontier.Next_V_Q()->ForAll(printBug, graph.nodes, util::DEVICE, stream));

	//std::cout << "Before CTA" << std::endl;
        std::cout << "queue_length = " << frontier.queue_length << std::endl;
        GUARD_CU(frontier.work_progress.GetQueueLength(
            frontier.queue_index, frontier.queue_length, true, stream,
            true));
	frontier.queue_length = total_length;
	//if (iteration == 0) frontier.queue_length = (VertexT) 2;
        //else if (iteration = 1) frontier.queue_length = (VertexT) 1;
        //else if (iteration = 2)  frontier.queue_length = (VertexT) 1;
        //else if (iteration = 3) frontier.queue_length = (VertexT) 0;
        std::cout << "queue_length " << frontier.queue_length << std::endl;

	}
    }

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &color_in = this->enactor->problem->data_slices[0][0].color_in;

    if (color_in) {
      auto &enactor_slices = this->enactor->enactor_slices;
      auto &iteration = enactor_slices[0].enactor_stats.iteration;
      // auto &frontiers = this->enactor->frontiers;
      auto &multistream_enactor_slices = this->enactor->multistream_enactor_slices;
      auto &num_stream = this->enactor->problem->num_stream;
      // Make sure there is no infinite loop - Disable if needed
      if (iteration >= 1000) return true;

      bool continue_predicate = false;
      for (int i = 0; i < num_stream; i++) {
        auto & frontier = multistream_enactor_slices[i].frontier;
        if (frontier.queue_length != 0) {
          continue_predicate |= true;
          break;
        }
      }
      return (!continue_predicate);
    }

    else {
      //DEBUG
	auto &enactor_slices = this->enactor->enactor_slices;
	auto &iteration = enactor_slices[0].enactor_stats.iteration;
      // if(iteration == 2) return true;  
      //std::cout << "Checking frontier length" << std::endl;
      auto &frontier = this->enactor->enactor_slices[0].frontier;
      if (frontier.queue_length == 0) return true;
      return false;
    }
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    // DEBUG
    printf("Executing Expand incoming ... \n");

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto iteration = enactor_slice.enactor_stats.iteration;
    auto &distances = data_slice.distances;
    auto &labels = data_slice.labels;
    auto &preds = data_slice.preds;
    auto label = this->enactor->mgpu_slices[this->gpu_num]
                     .in_iteration[iteration % 2][peer_];

    auto expand_op = [distances, labels, label, preds] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      ValueT in_val = value__associate_ins[in_pos];
      ValueT old_val = atomicMin(distances + key, in_val);
      if (old_val <= in_val) return false;
      if (labels[key] == label) return false;
      labels[key] = label;
      if (!preds.isEmpty()) preds[key] = vertex_associate_ins[in_pos];
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of SSSPIteration

/**
 * @brief SSSP enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::LabelT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  // Definations
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef typename Problem::LabelT LabelT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef SSSPIterationLoop<EnactorT> IterationT;

  // Multi-stream Specific - Public Attribute
  // typedef Frontier<VertexT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
  // FrontierT; FrontierT *frontiers = new FrontierT[num_stream]; cudaStream_t
  // *streams = new cudaStream_t[num_stream]; mgpu::ContextPtr *contexts = new
  // mgpu::ContextPtr[num_stream];
  typedef EnactorSlice<GraphT, LabelT, ARRAY_FLAG, cudaHostRegisterFlag>
                                   EnactorSliceT;
  util::Array1D<int, EnactorSliceT, ARRAY_FLAG,
                cudaHostRegisterFlag>  // | cudaHostAllocMapped |
                                       // cudaHostAllocPortable>
                                           multistream_enactor_slices;

  // Graph Coloring Specific - Public Attribute
  typedef color::Problem<GraphT> ColorProblemT;
  typedef color::Enactor<ColorProblemT> ColorEnactorT;
  ColorEnactorT color_enactor;

  // DEBUG
  VertexT path_src;

  // Members
  Problem *problem;
  IterationT *iterations;
  bool color_in;
  int num_stream;
  ValueT * h_distances;
  VertexT * h_preds;
  util::Array1D<SizeT, ValueT> * stream_distances;// = new util::Array1D<SizeT, ValueT > [num_stream];
  util::Array1D<SizeT, VertexT > * stream_preds;// = new util::Array1D<SizeT, VertexT > [num_stream];




  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief SSSPEnactor constructor
   */
  Enactor() : BaseEnactor("sssp"), color_enactor(), problem(NULL) {
    this->max_num_vertex_associates =
        (Problem::FLAG & Mark_Predecessors) != 0 ? 1 : 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief SSSPEnactor destructor
   */
  virtual ~Enactor() {
    // Release();
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    auto & num_stream = this->num_stream;
    for (int i = 0; i < num_stream; i++) {
      GUARD_CU(this->multistream_enactor_slices[i].Release(target));
    }
    GUARD_CU(this->multistream_enactor_slices.Release(target))

    printf("Done releasing multistream \n");

    // Graph Coloring Specific - Release
    if (this->color_in) {
      GUARD_CU(this->color_enactor.Release(target));
      printf("Done releasing color enactor \n");
      GUARD_CU(this->problem->color_problem.Release(target));
      printf("Done releasing color problem \n");
    }

    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;

    // Multi-stream specific - Release
    // if (this->frontiers != NULL && this->color_in) {
    //   for (int i = 0; i < num_stream; i++)
    //     GUARD_CU(this->frontiers[i].Release(target));
    //   delete[] this->frontiers;
    // }
    // if (this->streams != NULL && this->color_in) {
    //   for (int i = 0; i < num_stream; i++) {
    //     util::GRError(cudaStreamDestroy(this->streams[i]),
    //                   "cudaStreamDestroy failed.", __FILE__, __LINE__);
    //   }
    //   delete[] this->streams;
    // }
    return retval;
  }

  /**
   * @brief Initialize the enactor.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;
    this->color_in = problem.data_slices[0][0].color_in;
    this->num_stream = problem.num_stream;
    auto &num_stream = this->num_stream;
    //this->stream_distances = new util::Array1D<SizeT, ValueT > [num_stream];
    cudaMallocManaged(&(this->stream_distances), num_stream);
    //this->stream_preds = new util::Array1D<SizeT, VertexT > [num_stream];
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));

    // Multi-stream Specific - Init
    auto &multistream_enactor_slices = this->multistream_enactor_slices;
    multistream_enactor_slices.SetName("multistream_enactor_slices");
    GUARD_CU(multistream_enactor_slices.Allocate(num_stream, util::HOST));
    util::Parameters &parameters = problem.parameters;
    std::string advance_mode = parameters.Get<std::string>("advance-mode");
    std::string filter_mode  = parameters.Get<std::string>("filter-mode");
    int max_grid_size   = parameters.Get<int   >("max-grid-size");
    auto & data_slice = problem.data_slices[0][0];
    auto & distances = data_slice.distances;
    auto & preds = data_slice.preds;
    auto & graph = data_slice.sub_graph[0];

    //Set up memory for multi-stream
    this->h_distances = new ValueT[graph.nodes];
    this->h_preds = new VertexT[graph.nodes];

    //Create host side memory that will be update every iteration by multipler stream
    //GUARD_CU(distances.SetPointer(this->h_distances, graph.nodes, util::HOST));
    //GUARD_CU(preds.SetPointer(this->h_preds, graph.nodes, util::HOST));
    //distances.Move(util::HOST, util::DEVICE, graph.nodes, 0);
    //preds.Move(util::HOST, util::DEVICE, graph.nodes, 0);

    //Allocate space for each instantiation of distances and preds for each stream
    auto & stream_distances = this->stream_distances;
    //auto & stream_preds     = this->stream_preds;
    for (int i = 0; i < num_stream; i++) {
      stream_distances[i].SetName("distances " + std::to_string(i));
      //stream_preds[i].SetName("preds " + std::to_string(i));
      GUARD_CU(stream_distances[i] .Allocate(graph.nodes, util::HOST | util::DEVICE));
      //GUARD_CU(stream_preds[i] .Allocate(graph.nodes, util::DEVICE));
    }

    // Init multi-stream enactor slice
    for (int i = 0; i < num_stream; i++) {
      auto &multistream_enactor_slice = multistream_enactor_slices[i];
      GUARD_CU(multistream_enactor_slice.Init(
          2, NULL, "frontier[" + std::to_string(i) + "]", target,
          this->cuda_props + 0, advance_mode, filter_mode,
          max_grid_size));
      multistream_enactor_slice.oprtr_parameters.labels = &(problem.data_slices[0]->labels);
      auto &frontier = multistream_enactor_slice.frontier;
        frontier.Allocate(graph.nodes, graph.edges, this->queue_factors);
    }

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      // Coloring output frontier
      if (!(this->color_in)) {
        auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
        auto &graph = problem.sub_graphs[gpu];
        GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                                 this->queue_factors));
      }
      // Coloring input frontier
      if (this->color_in) {

        //This is work around, enactor slice is not actually used
        auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
        auto &graph = problem.sub_graphs[gpu];
        GUARD_CU(enactor_slice.frontier.Allocate(1, 1, this->queue_factors));

        for (int peer = 0; peer < this->num_gpus; peer++) {
          this->enactor_slices[gpu * this->num_gpus + peer]
              .oprtr_parameters.labels = &(problem.data_slices[gpu]->labels);
        }

      }
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));

    // If coloring input frontier, do coloring during enactor init
    if (this->color_in) {
      // Graph Coloring Specific - Init
      printf("Start Coloring Graph ... \n");
      util::CpuTimer cpu_timer;
      cpu_timer.Start();

      // Set up
      auto &graph = problem.sub_graphs[0];
      auto &color_problem = this->problem->color_problem;
      auto &color_enactor = this->color_enactor;
      GUARD_CU(color_problem.Init(graph, target));
      GUARD_CU(color_enactor.Init(color_problem, target));
      GUARD_CU(cudaDeviceSynchronize());

      // Reset
      printf("Initialize color enactor and problem \n");
      GUARD_CU(color_problem.Reset());
      GUARD_CU(cudaDeviceSynchronize());
      printf("Reset color problem \n");
      GUARD_CU(color_enactor.Reset());
      GUARD_CU(cudaDeviceSynchronize());

      // Enact
      printf("Reset color enactor and enact \n");
      GUARD_CU(color_enactor.Enact());
      GUARD_CU(cudaDeviceSynchronize());

      cpu_timer.Stop();

      // Report
      printf("Total color time: %f ms \n", cpu_timer.ElapsedMillis());
    }
    return retval;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    auto &num_stream = this->num_stream;
    // if coloring input frontier, reset the input frontiers according to color
    // sets
    if (this->color_in) {
      // Report colors
      auto &graph = this->problem->sub_graphs[0];
      printf("Get partitioned graph into frontiers\n");
      VertexT *colors = new VertexT[graph.nodes];
      GUARD_CU(this->problem->color_problem.Extract(colors));
      printf("Colors of 40 first nodes: \n");
      printf("[");
      for (int i = 0; i < 40; i++)
        printf("%d : %lld, ", i, (long long)colors[i]);
      printf("]\n");

      // Multi-stream Specific - Reset
      auto & stream_distances = this -> stream_distances;
      //auto & stream_preds = this -> stream_preds;
      auto & h_distances = this -> h_distances;
      auto & h_preds = this -> h_preds;
      auto & data_slice = this->problem->data_slices[0][0];
      auto & distances = data_slice.distances;
      auto & preds = data_slice.preds;

      // Sort independent sets
      int *tokens = new int[graph.nodes];
      int *frequencies = new int[graph.nodes];
      VertexT *pallette = new VertexT[graph.nodes];

      thrust::fill(thrust::host, tokens, tokens + graph.nodes, 1);

      // tokens is dummy vector for sort, sort colors by ascending order
      thrust::sort_by_key(thrust::host, colors, colors + graph.nodes, tokens);

      // DEBUG
      // for (int i = 0; i < graph.nodes; i++)
      //	printf("Ascending color: %d \n", colors[i]);

      // Find the top @num_stream colors
      thrust::pair<VertexT *, int *> new_end;
      new_end =
          thrust::reduce_by_key(thrust::host, colors, colors + graph.nodes,
                                tokens, pallette, frequencies);

      // DEBUG
      // for (int i = 0; i < 160; i++)
      //	printf("pallette[%d] = %d, freq[%d] = %d\n",i, pallette[i], i,
      // frequencies[i]);

      // Sort colors by frequencies
      int num_colors = new_end.first - pallette;
      thrust::sort_by_key(thrust::host, frequencies, frequencies + num_colors,
                          pallette, thrust::greater<int>());

      // Report top colors
      for (int i = 0; i < num_stream; i++)
        printf("The %d color is %d \n", i, pallette[i]);

      // Find length of each frontier
      int *lengths = new int[num_stream];
      int sum = 0;
      for (int i = 0; i < num_stream; i++) {
        // the last frontier has multiple colors
        if (i == num_stream - 1) {
          lengths[i] = thrust::reduce(thrust::host, &(frequencies[i]),
                                      &(frequencies[i]) + (num_colors - sum));
          break;
        }
        // the first @num_stream frontiers have 1 color each
        else {
          lengths[i] = frequencies[i];
          sum += 1;
        }
      }

      // Report size of frontiers
      for (int i = 0; i < num_stream; i++)
        printf("Frontier %d has length %d \n", i, lengths[i]);

      // Init frontiers
      // for (int i = 0; i < num_stream; i++) {
      //   GUARD_CU(
      //       this->frontiers[i].Init(2, NULL, std::to_string(i),
      //       util::DEVICE));
      //   util::GRError(cudaStreamCreate(&(this->streams[i])),
      //                 "cudaStreamCreate failed.", __FILE__, __LINE__);
      //   this->contexts[i] =
      //       mgpu::CreateCudaDeviceAttachStream(0, this->streams[i]);
      // }

      // Create sorted id list to populate frontiers;
      VertexT *id = new VertexT[graph.nodes];
      VertexT *sorted_id = new VertexT[graph.nodes];
#pragma omp parallel for
      for (int i = 0; i < graph.nodes; i++) id[i] = i;

      // Find nodes with the top color and put in sorted_id
      int offset = 0;
      for (int i = 0; i < num_stream - 1; i++) {
        // DEBUG
        printf("Copying color %d \n", i);
        auto pred_color = pallette[i];
        auto lambda = [colors, pred_color] __host__ __device__(
                          const VertexT v) { return colors[v] == pred_color; };
        if (i != 0) offset += lengths[i - 1];
        thrust::copy_if(thrust::host, id, id + graph.nodes, sorted_id + offset,
                        lambda);
      }

      // DEBUG
      printf("Copying last color \n");
      // Put the rest of the nodes in sorted_id
      auto lambda = [colors, pallette, num_stream] __host__ __device__(const VertexT v) {
        for (int i = 0; i < num_stream - 1; i++) {
          if (colors[v] == pallette[i]) return false;
        }
        return true;
      };
      offset += lengths[num_stream - 2];
      thrust::copy_if(thrust::host, id, id + graph.nodes, sorted_id + offset,
                      lambda);

      // DEBUG
      printf("All nodes has been moved and ready to populate frontier \n");
      // Intermediate variables, not used after this point
      delete[] pallette;
      delete[] tokens;
      delete[] frequencies;
      delete[] colors;
      delete[] id;

      // DEBUG
      printf("All intermediate values are deleted \n");
      // Alloc and populate frontiers
      offset = 0;
      auto &multistream_enactor_slices = this->multistream_enactor_slices;
      for (int i = 0; i < num_stream; i++) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        GUARD_CU(cudaDeviceSynchronize());
        // DEBUG
        printf("Populating frontier %d \n", i);
        if (i != 0) offset += lengths[i - 1];
        // this->frontiers[i].Allocate(lengths[i], graph.edges,
        //                             this->queue_factors);
        // GUARD_CU(cudaDeviceSynchronize());
        //
        // this->frontiers[i].queue_length = lengths[i];
        //
        // util::Array1D<SizeT, VertexT> tmp;
        // tmp.Allocate(lengths[i], util::DEVICE | util::HOST);
        // for (SizeT j = 0; j < lengths[i]; j++) {
        //   tmp[j] = (VertexT)(sorted_id + offset)[j];
        // }
        // GUARD_CU(tmp.Move(util::HOST, util::DEVICE));
        // GUARD_CU(this->frontiers[i].V_Q()->ForEach(
        //     tmp, [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
        //     lengths[i], util::DEVICE, 0));
        // tmp.Release();

        auto &frontier = multistream_enactor_slices[i].frontier;
        // frontier.Allocate(lengths[i], graph.edges, this->queue_factors);
        // frontier.Allocate(graph.nodes, graph.edges, this->queue_factors);
        // GUARD_CU(cudaDeviceSynchronize());

        frontier.queue_length = lengths[i];

        util::Array1D<SizeT, VertexT> tmp;
        // tmp.Allocate(lengths[i], util::DEVICE | util::HOST);
        // for (SizeT j = 0; j < lengths[i]; j++) {
        //   tmp[j] = (VertexT)(sorted_id + offset)[j];
        // }

        tmp.Allocate(graph.nodes, util::DEVICE | util::HOST);
        for (SizeT j = 0; j < graph.nodes; j++) {
          if (j < lengths[i])
            tmp[j] = (VertexT)(sorted_id + offset)[j];
          else
            tmp[j] = util::PreDefinedValues<VertexT>::InvalidValue;
        }
        GUARD_CU(tmp.Move(util::HOST, util::DEVICE));
        GUARD_CU(frontier.V_Q()->ForEach(
            tmp, [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
            lengths[i], util::DEVICE, 0));
        tmp.Release();
        GUARD_CU(cudaDeviceSynchronize());
      }

      delete[] sorted_id;


      //DEBUG
      printf("Done populating ... \n");

      // Move distances and preds from host to stream specific memory
      //for (int i = 0; i < num_stream; i++) {
        //GUARD_CU(stream_distances[i].EnsureSize_(graph.nodes, target));
        //GUARD_CU(stream_preds[i]   .EnsureSize_(graph.nodes, target));

        //GUARD_CU(stream_distances[i].SetPointer(h_distances, graph.nodes, util::HOST));
        //GUARD_CU(stream_preds[i] .SetPointer(h_preds, graph.nodes, util::HOST));
        // stream is needed for MemcpyAsync
        // auto & stream = multistream_enactor_slices[i].stream;
        // stream_distances[i].Move(util::DEVICE, util::HOST, graph.nodes, 0, stream);
        // stream_preds[i].Move(util::DEVICE, util::HOST, graph.nodes, 0, stream);
      //}

      this->thread_slices[0].init_size = 1;
    } //end if color_in

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      //DEBUG
      printf("Reseting token enactor slice of gpu %d \n",gpu);
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                [src] __host__ __device__(VertexT & v) { v = src; }, 1, target,
                0));
          }
        }
      }

      else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief one run of sssp, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        ((Enactor::Problem::FLAG & Mark_Predecessors) != 0) ? 1 : 0, 1,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a SSSP computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    this->path_src = src;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU SSSP Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace sssp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:


#include "simulation.cuh"
#include <vector>

SimulationRunner::SimulationRunner(int num_streams) : num_streams(num_streams) {
    streams.resize(num_streams);
    events.resize(num_streams);
    
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }
}

SimulationRunner::~SimulationRunner() {
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
}

void SimulationRunner::runSimulationStep(
    std::vector<NeuronGroup*>& neuron_groups,
    std::vector<SynapseGroup*>& synapse_groups) const {
    
    // 并行运行所有神经元组
    for (size_t i = 0; i < neuron_groups.size(); i++) {
        int stream_id = i % num_streams;
        neuron_groups[i]->run(streams[stream_id]);
    }
    
    // 等待所有神经元组完成计算
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // 并行运行所有突触组
    for (size_t i = 0; i < synapse_groups.size(); i++) {
        int stream_id = i % num_streams;
        synapse_groups[i]->run(streams[stream_id]);
    }
    
    // 等待所有突触组完成计算
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}

void SimulationRunner::runMultiStep(
    std::vector<NeuronGroup*>& neuron_groups,
    std::vector<SynapseGroup*>& synapse_groups,
    size_t steps) const {
    
    for (size_t step = 0; step < steps; step++) {
        runSimulationStep(neuron_groups, synapse_groups);
    }
}

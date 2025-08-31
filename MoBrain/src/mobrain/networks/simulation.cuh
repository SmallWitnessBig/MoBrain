#ifndef MOBRAIN_SIMULATION_CUH
#define MOBRAIN_SIMULATION_CUH

#include "neurons.cuh"
#include "synapses.cuh"
#include <vector>

class SimulationRunner {
private:
    int num_streams;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;

public:
    explicit SimulationRunner(int num_streams = 4);
    ~SimulationRunner();
    
    // 运行单个仿真步骤
    void runSimulationStep(
        std::vector<NeuronGroup*>& neuron_groups,
        std::vector<SynapseGroup*>& synapse_groups) const;
    
    // 运行多个仿真步骤
    void runMultiStep(
        std::vector<NeuronGroup*>& neuron_groups,
        std::vector<SynapseGroup*>& synapse_groups,
        size_t steps) const;
};

#endif // MOBRAIN_SIMULATION_CUH
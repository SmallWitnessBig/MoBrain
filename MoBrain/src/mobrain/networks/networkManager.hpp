//
// Created by 31530 on 2025/8/13.
//

#ifndef MOBRAIN_NETWORKMANAGER_HPP
#define MOBRAIN_NETWORKMANAGER_HPP
#include <unordered_map>
#include <utility>

#include "basic.cuh"
#include "neurons.cuh"
#include "synapses.cuh"
#include "simulation.cuh"
#include "input.cuh"
#include "output.hpp"
#include "utils/usefulMap.hpp"



class networkManager {
public:
    networkManager() = default;
    networkManager(const networkManager&) = delete;
    networkManager& operator=(const networkManager&) = delete;
    ~networkManager();
    posMap<std::unordered_map<Name,NeuronGroup*>> neuron_map;
    void addNeuronGroup(NeuronGroup* neuron_group, const Name &name);

    void addSynapseGroup(SynapseGroup* synapse_group);
    void removeSynapseGroup(SynapseGroup* synapse_group);

    void removeNeuronGroup(glm::vec3 pos,const Name& name);

    void run(const size_t steps) {
        simulationRunner.runMultiStep(neuronGroups, synapseGroups,steps);
    }

private:
    std::vector<NeuronGroup*> neuronGroups;
    std::vector<SynapseGroup*> synapseGroups;
    std::unordered_map<NeuronGroup*, std::vector<SynapseGroup*>> srcMap;
    std::unordered_map<NeuronGroup*, std::vector<SynapseGroup*>> dstMap;
    SimulationRunner simulationRunner;
};


#endif //MOBRAIN_NETWORKMANAGER_HPP
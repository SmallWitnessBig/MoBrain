//
// Created by 31530 on 2025/8/13.
//

#include "networkManager.hpp"

void networkManager::addNeuronGroup(NeuronGroup *neuron_group, const Name &name){
    neuron_map[neuron_group->pos][name]=neuron_group;
    neuronGroups.emplace_back(neuron_group);
}

void networkManager::addSynapseGroup(SynapseGroup* synapse_group) {
    synapseGroups.emplace_back(synapse_group);
    srcMap[synapse_group->src].emplace_back(synapse_group);
    dstMap[synapse_group->dst].emplace_back(synapse_group);
}
void networkManager::removeSynapseGroup(SynapseGroup* synapse_group) {
    synapseGroups.erase(std::ranges::find(synapseGroups, synapse_group));
    dstMap[synapse_group->dst].erase(std::ranges::find(dstMap[synapse_group->dst], synapse_group));
    srcMap[synapse_group->src].erase(std::ranges::find(srcMap[synapse_group->src], synapse_group));
    delete synapse_group;
}
void networkManager::removeNeuronGroup(const glm::vec3 pos,const Name& name)  {
    NeuronGroup* neuron_group = neuron_map[pos][name];
    if (neuron_map[pos].size()==1) {
        neuron_map.erase(pos);
    }else {
        neuron_map[pos].erase(name);
    }
    neuronGroups.erase(std::ranges::find(neuronGroups, neuron_group));
    for (const auto& i: srcMap[neuron_group]) {
        synapseGroups.erase(std::ranges::find(synapseGroups, i));
        delete i;
    }
    for (const auto& i: dstMap[neuron_group]) {
        synapseGroups.erase(std::ranges::find(synapseGroups, i));
        delete i;
    }
    dstMap.erase(neuron_group);
    srcMap.erase(neuron_group);
    delete neuron_group;

}

networkManager::~networkManager() {
    neuron_map.clear();
    dstMap.clear();
    srcMap.clear();

    for (auto& i : neuronGroups) {
        delete i;
    }
    for (auto& i : synapseGroups) {
        delete i;
    }
}



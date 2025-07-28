#pragma once
#include "vulkaninit.hpp"
class QueueFamilyIndices {
public:
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() const {
        return graphicsFamily.has_value();
    }
};
QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice);
void pickPhysicalDevice();
void createLogicalDevice();

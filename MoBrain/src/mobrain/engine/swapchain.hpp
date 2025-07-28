#pragma once
#include "vulkaninit.hpp"
#include "../core/context.hpp"

class SwapChainSupportDetails {
public:
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR>  formats;
    std::vector<vk::PresentModeKHR> presentModes;

};
SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& physicalDevice);
void createSwapChain();
void recreateSwapChain();

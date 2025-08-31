#pragma once
#include "vulkaninit.hpp"
#include "../core/context.hpp"
/**
*   基本表面功能（交换链中图像的最小/最大数量，图像的最小/最大宽度和高度）
*   可用表面格式（像素格式，色彩空间）
*   可用呈现模式
*/
class SwapChainSupportDetails {
public:
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR>  formats;
    std::vector<vk::PresentModeKHR> presentModes;

};
SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& physicalDevice);
void createSwapChain();
void recreateSwapChain();

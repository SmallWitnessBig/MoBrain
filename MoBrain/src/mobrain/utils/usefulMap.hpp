//
// Created by 31530 on 2025/8/22.
//

#ifndef MOBRAIN_USEFULMAP_HPP
#define MOBRAIN_USEFULMAP_HPP
#include <glm/glm.hpp>
#include <functional>
#include <unordered_map>
namespace std {
    template<> struct hash<glm::vec3> {
        size_t operator()(const glm::vec3& v) const {
            return hash<float>()(v.x) ^ (hash<float>()(v.y)<<1) ^ (hash<float>()(v.z)<<2);
        }
    };
}
template <class T>
using posMap = std::unordered_map<glm::vec3,T>;


#endif //MOBRAIN_USEFULMAP_HPP
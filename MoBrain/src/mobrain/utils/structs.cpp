#include "structs.hpp"
#include "../engine/buffers.hpp"
vk::VertexInputBindingDescription Vertex::getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription;
	bindingDescription
        .setBinding(0)
        .setInputRate(vk::VertexInputRate::eVertex)
        .setStride(sizeof(Vertex));
    return bindingDescription;
}

std::array<vk::VertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
    
    attributeDescriptions[0]
        .setBinding(0)
        .setLocation(0)
        .setFormat(vk::Format::eR32G32B32Sfloat)
        .setOffset(offsetof(Vertex, pos));
        
    attributeDescriptions[1]
        .setBinding(0)
        .setLocation(1)
        .setFormat(vk::Format::eR32G32B32Sfloat)
        .setOffset(offsetof(Vertex, color));
        
    return attributeDescriptions;
}
Cube::Cube(glm::vec3 _pos, glm::vec3 _color)
    :Object(_pos, _color)
{
    index_count = 36;
    indices = {
        2, 1, 0, 0, 3, 2,
        4, 5, 6, 6, 7, 4,
        2, 3, 7, 7, 6, 2,
        0, 1, 5, 5, 4, 0,
        1, 2, 6, 6, 5, 1,
        0, 4, 7, 7, 3, 0
    };
    vertices = {
        {glm::vec3(-0.5f, -0.5f,-0.5f)+_pos,  _color},
        {glm::vec3(0.5f, -0.5f,-0.5f)+_pos,  _color},
        {glm::vec3(0.5f, 0.5f,-0.5f)+_pos,  _color},
        {glm::vec3(-0.5f, 0.5f,-0.5f)+_pos,  _color},
        {glm::vec3(-0.5f, -0.5f,0.5f)+_pos,  _color},
        {glm::vec3(0.5f, -0.5f,0.5f)+_pos, _color},
        {glm::vec3(0.5f, 0.5f,0.5f)+_pos,  _color},
        {glm::vec3(-0.5f, 0.5f,0.5f)+_pos,  _color}
    };
}
Cube& Cube::draw(const vk::raii::CommandBuffer& _cmb) {

    return *this;
}
ObjectType Cube::getType() {
    return ObjectType::CUBE;
}
Scene& Scene::addObject(std::shared_ptr<Object> _object) {
    objects.push_back(_object);
    positions.push_back(_object->pos);
    return *this;
}
Scene& Scene::removeObject(std::shared_ptr<Object> _object) {
    positions.erase(std::find(positions.begin(), positions.end(), _object->pos));
    objects.erase(std::find(objects.begin(), objects.end(), _object));
    return *this;
}
Scene& Scene::removeObject(glm::vec3 _pos) {
    positions.erase(std::find(positions.begin(), positions.end(), _pos));
    objects.erase(std::find_if(objects.begin(), objects.end(), [&](std::shared_ptr<Object> obj) {
        return obj->pos == _pos;
    }));
    return *this;
}
Role& Role::update() {
    glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f);
    for (size_t t = 0; t < 5; t++) {
        if (app.render_scene.objects(pos=glm::vec3{
                round(camera::pos.x + camera::front.x * t),
                round(camera::pos.y + camera::front.y * t),
                round(camera::pos.z + camera::front.z * t)}) == 1) 
        {
            reach.clear();
            reach.emplace_back(pos,glm::vec3{1.0,1.0,1.0});
            return *this;
        }
    }
    return *this;
}

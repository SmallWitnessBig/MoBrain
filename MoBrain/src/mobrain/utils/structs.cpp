#include "structs.hpp"
#include "../engine/buffers.hpp"
void Scene::addCube(const Cube* cube) {
    if (pos_map.find(cube->pos) != pos_map.end()) {
       return;
    }
	cubes.emplace_back(cube);
    cube_map[cube] = cubes.size() - 1;
    pos_map[cube->pos] = cubes.size() - 1;
	app.bufferM.addCube(cube);
}

void Scene::removeCube(const Cube* cube) {
    if (!cube || cube_map.find(cube) == cube_map.end()) return;
    size_t index = cube_map[cube];
    app.bufferM.removeCube(cube);
    cube_map.erase(cube);
    pos_map.erase(cube->pos);
    delete cube;
    cubes.erase(cubes.begin() + index);
    for (size_t i = index; i < cubes.size(); i++) {
        cube_map[cubes[i]] = i;
        pos_map[cubes[i]->pos] = i;
    }
}

Scene& Scene::initScene() {
	for (float x = -25.0f; x < 25.0f; x += 1.0f) {
		for (float y = -25.0f; y < 25.0f; y += 1.0f) {
            Cube* cube = new Cube(glm::vec3{ x,0,y }, glm::vec4{ x * x / 625.0f,y * y / 625.0f,x * x / 625.0f,1.0f });
			addCube(cube);
		}
	}
	return *this;
}

Scene::~Scene() {
	for (auto& cube : cubes) {
		delete cube;
	}
    cubes.clear();
    cube_map.clear();
    pos_map.clear();
}









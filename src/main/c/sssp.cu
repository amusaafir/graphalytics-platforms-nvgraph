#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include "nvgraph.h"

typedef struct COO_List coo_list;
typedef struct CSR_List csr_list;

COO_List* load_graph_from_edge_list_file_to_coo(std::vector<int>&, std::vector<int>&, std::vector<float>&, char*);
void print_output(float *results, int nvertices);
void print_csr(int*, int*);
void print_csc(int*, int*);
void print_coo(int*, int*, float*);

typedef struct COO_List {
	int* source;
	int* destination;
	float* edge_data;
} COO_List;

typedef struct CSR_List {
	int* offsets;
	int* indices;
} CSR_List;

typedef struct CSC_List {
    int* destination_offsets;
    int* source_indices;
} CSC_List;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

int SIZE_VERTICES;
int SIZE_EDGES;

std::string getEpoch() {
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count());
}

int add_vertex_as_coordinate(std::vector<int>& vertices_type, std::unordered_map<int, int>& map_from_edge_to_coordinate, int vertex, int coordinate) {
	if (map_from_edge_to_coordinate.count(vertex)) {
		vertices_type.push_back(map_from_edge_to_coordinate.at(vertex));

		return coordinate;
	}
	else {
		map_from_edge_to_coordinate[vertex] = coordinate;
		vertices_type.push_back(coordinate);
		coordinate++;

		return coordinate;
	}
}

COO_List* load_graph_from_edge_list_file_to_coo(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, std::vector<float>& edge_data, char* file_path) {
	printf("\nLoading graph file from: %s", file_path);

	FILE* file = fopen(file_path, "r");

	char line[256];

	int current_coordinate = 0;

    std::unordered_map<int, int> map_from_edge_to_coordinate;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        // Save source and target vertex (temp)
        int source_vertex;
        int target_vertex;
        float edge_weight;

        sscanf(line, "%d%d%f\t", &source_vertex, &target_vertex, &edge_weight);

        // Add vertices to the source and target arrays, forming an edge accordingly
        current_coordinate = add_vertex_as_coordinate(source_vertices, map_from_edge_to_coordinate, source_vertex, current_coordinate);
        current_coordinate = add_vertex_as_coordinate(destination_vertices, map_from_edge_to_coordinate, target_vertex, current_coordinate);
        edge_data.push_back(edge_weight);
    }

    SIZE_VERTICES = map_from_edge_to_coordinate.size();
    SIZE_EDGES = source_vertices.size();

    printf("\nTotal amount of vertices: %zd", SIZE_VERTICES);
    printf("\nTotal amount of edges: %zd", SIZE_EDGES);
    printf("\nData:");

    for (int i = 0 ; i<edge_data.size(); i++) {
        printf("\n%f", edge_data[i]);
    }

	COO_List* coo_list = (COO_List*)malloc(sizeof(COO_List));

	source_vertices.reserve(source_vertices.size());
	destination_vertices.reserve(destination_vertices.size());
	edge_data.reserve(edge_data.size());
	coo_list->source = &source_vertices[0];
	coo_list->destination = &destination_vertices[0];
	coo_list->edge_data = &edge_data[0];

	if (source_vertices.size() != destination_vertices.size()) {
		printf("\nThe size of the source vertices does not equal the destination vertices.");
		exit(1);
	}

	// Print edges
	/*for (int i = 0; i < source_vertices.size(); i++) {
	printf("\n(%d, %d)", coo_list->source[i], coo_list->destination[i]);
	}*/

	fclose(file);

	return coo_list;
}

void convert_coo_to_csc_format(int* source_indices_h, int* destination_indices_h, float* edge_data_h) {
    print_coo(source_indices_h, destination_indices_h, edge_data_h);

    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    cudaDataType_t edge_dimT = CUDA_R_32F;

    //specifics for the dataset being used
    int nvertices = SIZE_VERTICES, nedges = SIZE_EDGES; //REMEMBER TO CHANGE THIS FOR EVERY NEW FILE
    //int *source_indices_h, *destination_indices_h;
    //float *edge_data_h, *bookmark_h;
    //source_indices_h = (int *)malloc(sizeof(int)*nedges);
    //destination_indices_h = (int *)malloc(sizeof(int)*nedges);
    //edge_data_h = (float *)malloc(sizeof(float)*nedges);
    //bookmark_h = (float*)malloc(sizeof(float)*nvertices);

    //initialization of nvGraph
    nvgraphCreate(&handle);
    nvgraphCreateGraphDescr(handle, &graph);

    nvgraphCSCTopology32I_t col_major_topology = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));




    cudaDataType_t data_type = CUDA_R_32F;
    //Initialize unconverted topology
    nvgraphCOOTopology32I_t current_topology = (nvgraphCOOTopology32I_t)malloc(sizeof(struct nvgraphCOOTopology32I_st));
    current_topology->nedges = nedges;
    current_topology->nvertices = nvertices;
    current_topology->tag = NVGRAPH_UNSORTED; //NVGRAPH_UNSORTED, NVGRAPH_SORTED_BY_SOURCE or NVGRAPH_SORTED_BY_DESTINATION can also be used

    cudaMalloc((void**)&(current_topology->destination_indices), nedges*sizeof(int));
    cudaMalloc((void**)&(current_topology->source_indices), nedges*sizeof(int));

    //Copy data into topology
    cudaMemcpy(current_topology->destination_indices, destination_indices_h, nedges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(current_topology->source_indices, source_indices_h, nedges*sizeof(int), cudaMemcpyHostToDevice);

    //Allocate and copy edge data
    float *edge_data_d, *dst_edge_data_d;
    cudaMalloc((void**)&edge_data_d, nedges*sizeof(float));
    cudaMalloc((void**)&dst_edge_data_d, nedges*sizeof(float));
    cudaMemcpy(edge_data_d, edge_data_h, nedges*sizeof(float), cudaMemcpyHostToDevice);

    int *indices_h, *offsets_h, **indices_d, **offsets_d;
    //These are needed for compiler issues (the possibility that the initialization is skipped)
    nvgraphCSCTopology32I_t csc_topology;

    csc_topology = (nvgraphCSCTopology32I_t) col_major_topology;
    indices_d = &(csc_topology->source_indices);
    offsets_d = &(csc_topology->destination_offsets);

    cudaMalloc((void**)(indices_d), nedges*sizeof(int));
    cudaMalloc((void**)(offsets_d), (nvertices + 1)*sizeof(int));
    indices_h = (int*)malloc(nedges*sizeof(int));
    offsets_h = (int*)malloc((nvertices + 1)*sizeof(int));

    check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, current_topology, edge_data_d, &data_type, NVGRAPH_CSC_32, col_major_topology, dst_edge_data_d));

    //Copy converted topology from device to host
    cudaMemcpy(indices_h, *indices_d, nedges*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(offsets_h, *offsets_d, (nvertices + 1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(edge_data_h, dst_edge_data_d, nedges*sizeof(float), cudaMemcpyDeviceToHost);

    print_csc(offsets_h, indices_h, edge_data_h);

    //bookmark..
    //free mem..
    //copy data back to host

}

int main(int argc, char **argv) {
    if (argc == 2) {
        std::vector<int> source_vertices;
        std::vector<int> destination_vertices;
        std::vector<float> edge_data;

        COO_List* coo_list = load_graph_from_edge_list_file_to_coo(source_vertices, destination_vertices, edge_data, argv[1]);

        // Convert the COO graph into a CSR format (for the in-memory GPU representation)
        /*CSC_List* csc_list = */convert_coo_to_csc_format(coo_list->source, coo_list->destination, coo_list->edge_data);
        //print_csc(csc_list->destination_offsets, csc_list->source_indices);
    } else {
        std::cout<< "Woops: Incorrect nr/values of input params.";
    }
    /*
    const size_t  n = 6, nnz = 10, vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status; nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
    float weights_h[] = {0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5};
    int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};

    check(nvgraphCreate(&handle));

    check(nvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::cout << "Processing starts at: " << getEpoch() << std::endl;

    // Solve
    int source_vert = 0;
    check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));

    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Processing ends at: " << getEpoch() << std::endl;

    if (argc == 2) {
        std::ofstream myfile;
        myfile.open(argv[1]);
        myfile << "2 0.000000000000000e+00\n3 8.200000000000000e-01\n4 6.899999999999999e-01\n5 1.260000000000000e+00\n6 1.780000000000000e+00\n7 2.310000000000000e+00\n8 1.140000000000000e+00\n9 2.010000000000000e+00\n10 2.410000000000000e+00";
        myfile.close();
    } else {
        std::cout<"WOOPS: no params provided? ";
    }

    // Clean
    print_output(sssp_1_h, 6);
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    */
    return 0;
}

void print_output(float *results, int nvertices) {
    for (int i = 0; i < nvertices; i++) {
        printf("%f \n", results[i]);
    }
}

void print_csr(int* h_offsets, int* h_indices) {
	printf("\nRow Offsets (Vertex Table):\n");
	for (int i = 0; i < SIZE_VERTICES + 1; i++) {
		printf("%d, ", h_offsets[i]);
	}

	printf("\nColumn Indices (Edge Table):\n");
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("%d, ", h_indices[i]);
	}
}

void print_csc(int* d_offsets, int* s_indices, float* weight) {
	printf("\nRow Offsets (Vertex Table):\n");
	for (int i = 0; i < SIZE_VERTICES + 1; i++) {
		printf("%d, ", d_offsets[i]);
	}

	printf("\nColumn Indices (Edge Table):\n");
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("%d, ", s_indices[i]);
	}

	printf("\nEdge weight:\n");
	for (int i = 0; i < SIZE_EDGES; i++) {
        printf("%f, ", weight[i]);
    }

}


void print_coo(int* source, int* target, float* weight) {
   for (int i = 0 ; i<SIZE_EDGES; i++) {
        printf("\n(%d,%d) - %f", source[i], target[i], weight[i]);
   }
}
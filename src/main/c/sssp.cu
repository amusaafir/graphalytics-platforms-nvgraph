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

#include <unordered_map>
#include <unordered_set>

#include <stdlib.h>


typedef struct COO_List coo_list;
typedef struct CSR_List csr_list;

void print_output(float *results, int nvertices);
void print_csr(int*, int*);
void print_csc(int*, int*, float*);
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
    float* edge_data;
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
int IS_GRAPH_UNDIRECTED;
int SSSP_SOURCE_VERTEX = -1;
std::unordered_map<int, long long> map_from_coordinate_to_vertex; // Required for validation

std::string getEpoch() {
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count());
}

int add_vertex_as_coordinate(std::vector<int>& vertices_type, std::unordered_map<long long, int>& map_from_vertex_to_coordinate, std::unordered_map<int, long long>& map_from_coordinate_to_vertex, long long vertex, int coordinate) {
    if (map_from_vertex_to_coordinate.count(vertex)) {
        vertices_type.push_back(map_from_vertex_to_coordinate.at(vertex));

        return coordinate;
    } else {
        map_from_vertex_to_coordinate[vertex] = coordinate;
        vertices_type.push_back(coordinate);

    	if (vertex == SSSP_SOURCE_VERTEX) {
    	     SSSP_SOURCE_VERTEX = coordinate;	
             printf("\n The source COO vertex = %d\n",  coordinate);
    	}

        map_from_coordinate_to_vertex[coordinate] = vertex;

        coordinate++;

        return coordinate;
    }
}

void save_input_file_as_coo(std::vector<int>& source_vertices_vect, std::vector<int>& destination_vertices_vect, std::vector<float>& edge_data_vect, char* save_path) {
    printf("\nWriting results to output file.");

    char* file_path = save_path;
    FILE *output_file = fopen(file_path, "w");

    if (output_file == NULL) {
        printf("\nError writing results to output file.");
        exit(1);
    }

    for (int i = 0; i < source_vertices_vect.size(); i++) {
        fprintf(output_file, "%d\t%d\t%f\n", source_vertices_vect[i], destination_vertices_vect[i], edge_data_vect[i]);
    }

    fclose(output_file);
}

void save_bfs_result(int* result, char* save_path) {
    printf("\nWriting BFS results to output file.");

    char* file_path = save_path;
    FILE *output_file = fopen(file_path, "w");

    if (output_file == NULL) {
        printf("\nError writing results to output file.");
        exit(1);
    }

    for (int i = 0; i < SIZE_VERTICES; i++) {
        if (result[i] >= std::numeric_limits<int>::max()) {
         fprintf(output_file, "%lu %d\n", map_from_coordinate_to_vertex[i], "inf");
         continue;
        }
        //printf("yay: %lu %d  %d \n", map_from_coordinate_to_vertex[i], i,  result[i]);
        fprintf(output_file, "%lu %d\n", map_from_coordinate_to_vertex[i], result[i]);
    }

    fclose(output_file);
}

void save_sssp_result(float* result, char* save_path) {
    printf("\nWriting SSSP results to output file.");

    char* file_path = save_path;
    FILE *output_file = fopen(file_path, "w");

    if (output_file == NULL) {
        printf("\nError writing results to output file.");
        exit(1);
    }

    for (int i = 0; i < SIZE_VERTICES; i++) {
         
          if (result[i] >= std::numeric_limits<float>::max()) {
            fprintf(output_file, "%lu %s\n", map_from_coordinate_to_vertex[i], "inf");
             continue;
          } 
          fprintf(output_file, "%lu %f\n", map_from_coordinate_to_vertex[i], result[i]);
    }

    fclose(output_file);
}

COO_List* load_graph_from_edge_list_file_to_coo(std::vector<int> source_vertices_vect, std::vector<int> destination_vertices_vect, std::vector<float> edge_data_vect, char* file_path) {
    printf("\nLoading graph file from: %s to COO", file_path);


    FILE* file = fopen(file_path, "r");

    char line[256];

    int current_coordinate = 0;

    std::unordered_map<long long, int> map_from_vertex_to_coordinate;




    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') {
            //print_debug_log("\nEscaped a comment.");
            continue;
        }

        // Save source, destination and weight
        long long source_vertex;
        long long target_vertex;
        float weight;

        sscanf(line, "%lu%lu%f\t", &source_vertex, &target_vertex, &weight);

        // Add vertices to the source and target arrays, forming an edge accordingly
        current_coordinate = add_vertex_as_coordinate(source_vertices_vect, map_from_vertex_to_coordinate, map_from_coordinate_to_vertex, source_vertex, current_coordinate);
        current_coordinate = add_vertex_as_coordinate(destination_vertices_vect, map_from_vertex_to_coordinate, map_from_coordinate_to_vertex, target_vertex, current_coordinate);

        edge_data_vect.push_back(weight);


        // Add the end vertices but swap places this time (undirected)
        if (IS_GRAPH_UNDIRECTED) {
            current_coordinate = add_vertex_as_coordinate(source_vertices_vect, map_from_vertex_to_coordinate, map_from_coordinate_to_vertex, target_vertex, current_coordinate);
            current_coordinate = add_vertex_as_coordinate(destination_vertices_vect, map_from_vertex_to_coordinate, map_from_coordinate_to_vertex, source_vertex, current_coordinate);

            edge_data_vect.push_back(weight);
        }
    }

    fclose(file);
/*
printf("The source vertices vect size is\n");


for(auto it = map_from_vertex_to_coordinate.cbegin(); it != map_from_vertex_to_coordinate.cend(); ++it)
{
    std::cout << it->first << " " << it->second << "\n";
   //printf("/n%lu %lu %f", it->first, it->second);
}

printf("---------");
*/

    SIZE_VERTICES = map_from_vertex_to_coordinate.size();
    SIZE_EDGES = source_vertices_vect.size();

    printf("\nTotal amount of vertices: %zd", SIZE_VERTICES);
    printf("\nTotal amount of edges: %zd", SIZE_EDGES);

    if (source_vertices_vect.size() != destination_vertices_vect.size()) {
        printf("\nThe size of the source vertices does not equal the destination vertices.");
        exit(1);
    }

//    save_input_file_as_coo(source_vertices_vect, destination_vertices_vect,edge_data_vect, "/var/scratch/musaafir/graphalytics-platforms-nvgraph/src/main/c/coo_save.txt");

    //int* source_vertices = &source_vertices_vect[0];
    //int* destination_vertices = &destination_vertices_vect[0];
    //float* edge_data = &edge_data_vect[0];

    COO_List* coo_list = (COO_List*)malloc(sizeof(COO_List));

    //source_vertices_vect.reserve(source_vertices_vect.size());
    //destination_vertices_vect.reserve(destination_vertices_vect.size());
    //edge_data_vect.reserve(edge_data_vect.size());



//    coo_list->source = &source_vertices_vect[0];
  //  coo_list->destination = &destination_vertices_vect[0];
   // coo_list->edge_data = &edge_data_vect[0];
    /*
    for (std::unordered_map<int, int>::const_iterator it = map_from_coordinate_to_vertex.begin(); it != map_from_coordinate_to_vertex.end(); it++) {
    	printf("coordinate: %d, vertex: %d\n", it->first, it->second);
	}
*//*
    printf("Printing source & destination edges (vect)");

    for (int i = 0; i < SIZE_EDGES; i++) {
        printf("\n(%d, %d - %f)", source_vertices_vect[i], destination_vertices_vect[i], edge_data_vect[i]);
    }

*/

    int* source_arr = (int*) malloc(sizeof(int) * SIZE_EDGES);
    int* destination_arr = (int*) malloc(sizeof(int) * SIZE_EDGES);
    float* edge_data_arr = (float*) malloc(sizeof(float) * SIZE_EDGES);

    for (int i = 0; i < SIZE_EDGES; i++) {
              source_arr[i] = source_vertices_vect[i];
    destination_arr[i] = destination_vertices_vect[i];
    edge_data_arr[i] = edge_data_vect[i];

//             printf("\n(%d, %d - %f)", source_vertices_vect[i], destination_vertices_vect[i], edge_data_vect[i]);
    }

    coo_list->source = source_arr;
    coo_list->destination = destination_arr;
    coo_list->edge_data = edge_data_arr;


    return coo_list;
}

CSC_List* convert_coo_to_csc_format(int* source_indices_h, int* destination_indices_h, float* edge_data_h) {
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    cudaDataType_t edge_dimT = CUDA_R_32F;

    //specifics for the dataset being used
    int nvertices = SIZE_VERTICES, nedges = SIZE_EDGES;

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
    //printf("\nCHECK 1");
    cudaMalloc((void**)&(current_topology->destination_indices), nedges*sizeof(int));
  
    //printf("\nCHECK 2");
    cudaMalloc((void**)&(current_topology->source_indices), nedges*sizeof(int));
    //printf("\ncheck 3");
    //Copy data into topology
    cudaMemcpy(current_topology->destination_indices, destination_indices_h, nedges*sizeof(int), cudaMemcpyHostToDevice);
    //printf("\nCheck 4");
    cudaMemcpy(current_topology->source_indices, source_indices_h, nedges*sizeof(int), cudaMemcpyHostToDevice);
    //printf("\ncheck 5");
    //Allocate and copy edge data
    float *edge_data_d, *dst_edge_data_d;
    cudaMalloc((void**)&edge_data_d, nedges*sizeof(float));
    //printf("\nCheck 6");
    cudaMalloc((void**)&dst_edge_data_d, nedges*sizeof(float));
    cudaMemcpy(edge_data_d, edge_data_h, nedges*sizeof(float), cudaMemcpyHostToDevice);

    int *indices_h, *offsets_h, **indices_d, **offsets_d;
    //These are needed for compiler issues (the possibility that the initialization is skipped)
    nvgraphCSCTopology32I_t csc_topology;

    csc_topology = (nvgraphCSCTopology32I_t) col_major_topology;
    indices_d = &(csc_topology->source_indices);
    offsets_d = &(csc_topology->destination_offsets);
    //printf("\ncheck 7");
    cudaMalloc((void**)(indices_d), nedges*sizeof(int));
   
    //printf("\ncheck 8");
    cudaMalloc((void**)(offsets_d), (nvertices + 1)*sizeof(int));
    indices_h = (int*)malloc(nedges*sizeof(int));
    offsets_h = (int*)malloc((nvertices + 1)*sizeof(int));
    //printf("\ncheck 8.3");
    check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, current_topology, edge_data_d, &data_type, NVGRAPH_CSC_32, col_major_topology, dst_edge_data_d));
    //printf("check 8.5");
    //Copy converted topology from device to host
    cudaMemcpy(indices_h, *indices_d, nedges*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(offsets_h, *offsets_d, (nvertices + 1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(edge_data_h, dst_edge_data_d, nedges*sizeof(float), cudaMemcpyDeviceToHost);
    //printf("\ncheck 9");

    CSC_List* csc_list = (CSC_List*)malloc(sizeof(CSC_List));
    csc_list->destination_offsets = offsets_h;
    csc_list->source_indices = indices_h;
    csc_list->edge_data = edge_data_h;

    cudaFree(indices_d);
    cudaFree(offsets_d);
    cudaFree(dst_edge_data_d);
    cudaFree(edge_data_d);
    cudaFree(current_topology->source_indices);
    cudaFree(current_topology->destination_indices);
    free(current_topology);

    //bookmark..
    //copy data back to host

    return csc_list;
}

int* bfs(int* source_offsets_h, int* destination_indices_h) {
    std::chrono::steady_clock::time_point startMakespan = std::chrono::steady_clock::now();
    std::cout << "Makespan starts at: " << getEpoch() << std::endl;

    //Example of graph (CSR format)
    const size_t  n = SIZE_VERTICES, nnz = SIZE_EDGES, vertex_numsets = 2, edge_numset = 0;
    //where to store results (distances from source) and where to store results (predecessors in search tree)
    int bfs_predecessors_h[n];
    int* bfs_distances_h = (int*) malloc(SIZE_VERTICES*sizeof(int));
    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;
    cudaDataType_t* vertex_dimT;
    size_t distances_index = 0;
    size_t predecessors_index = 1;
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    vertex_dimT[distances_index] = CUDA_R_32I;
    vertex_dimT[predecessors_index] = CUDA_R_32I;
    //Creating nvgraph objects
    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    // Set graph connectivity and properties (tranfers)
    CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    CSR_input->nvertices = SIZE_VERTICES;
    CSR_input->nedges = SIZE_EDGES;
    CSR_input->source_offsets = source_offsets_h;
    CSR_input->destination_indices = destination_indices_h;
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    int source_vert = SSSP_SOURCE_VERTEX;
    //Setting the traversal parameters
    nvgraphTraversalParameter_t traversal_param;
    nvgraphTraversalParameterInit(&traversal_param);
    nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
    nvgraphTraversalSetPredecessorsIndex(&traversal_param, predecessors_index);
    nvgraphTraversalSetUndirectedFlag(&traversal_param, false);
    //Computing traversal using BFS algorithm
    check_status(nvgraphTraversal(handle, graph, NVGRAPH_TRAVERSAL_BFS, &source_vert, traversal_param));
    // Get result
    check_status(nvgraphGetVertexData(handle, graph, (void*) bfs_distances_h, distances_index));
    check_status(nvgraphGetVertexData(handle, graph, (void*) bfs_predecessors_h, predecessors_index));
    // expect bfs distances_h = (1 0 1 3 3 2 2147483647)
    //for (int i = 0; i<n; i++)  printf("Distance to vertex %d: %i\n",i, bfs_distances_h[i]); printf("\n");
    // expect bfs predecessors = (1 -1 1 5 5 0 -1)
    //for (int i = 0; i<n; i++)  printf("Predecessor of vertex %d: %i\n",i, bfs_predecessors_h[i]); printf("\n");
    free(vertex_dimT);
    free(CSR_input);
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy(handle));

    std::chrono::steady_clock::time_point endMakespan = std::chrono::steady_clock::now();
    std::cout << "Makespan ends at: " << getEpoch() << std::endl;

    return bfs_distances_h;
}

float* sssp(int* source_indices, int* destination_offsets, float* weights) {
    //printf("\nPerforming SSSP");
    std::chrono::steady_clock::time_point startMakespan = std::chrono::steady_clock::now();
    std::cout << "Makespan starts at: " << getEpoch() << std::endl;
    const size_t  n = SIZE_VERTICES, nnz = SIZE_EDGES, vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;
    // nvgraph variables
    nvgraphStatus_t status; nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    // Init host data
    //printf("\nCHECK 1000");
    sssp_1_h = (float*)malloc(n*sizeof(float));
    //printf("\nCHECK 2000");
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    //printf("\nCHECK 3000");
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    //printf("\nCHECK 4000");
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
    //float weights_h[] = {0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5};
    //int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    //int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};
    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets;
    CSC_input->source_indices = source_indices;
    // Set graph connectivity and properties (tranfers)
    //printf("\nCHECK 5000");
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    //printf("\nCHECK 5001");
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    //printf("\nCHECK 5002");
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    //printf("\n CHECK 5003");
    check(nvgraphSetEdgeData(handle, graph, (void*)weights, 0));
    // Solve
    //printf("\nCHECK 6000");

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::cout << "Processing starts at: " << getEpoch() << std::endl;

    check(nvgraphSssp(handle, graph, 0,  &SSSP_SOURCE_VERTEX, 0));

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Processing ends at: " << getEpoch() << std::endl;

    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    //printf("\nDone with sssp");
    //Clean
    //free(sssp_1_h);
    free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));

    std::chrono::steady_clock::time_point endMakespan = std::chrono::steady_clock::now();
    std::cout << "Makespan ends at: " << getEpoch() << std::endl;

    return sssp_1_h;
}


int main(int argc, char **argv) {
    IS_GRAPH_UNDIRECTED = strtol(argv[3], NULL, 10);
    SSSP_SOURCE_VERTEX = strtol(argv[4], NULL, 10);
    char* SELECTED_ALGORITHM = argv[5];
    std::cout << "Input graph path: " << argv[1] << "\n";
    std::cout << "Result output path: " << argv[2] << "\n";
    std::cout << "Is undirected: " << IS_GRAPH_UNDIRECTED << "\n";
    std::cout << "Source vertex: " << SSSP_SOURCE_VERTEX << "\n";
    std::cout << "Algorithm: " << SELECTED_ALGORITHM << "\n";

    std::vector<int> source_vertices_vect;
    std::vector<int> destination_vertices_vect;
    std::vector<float> edge_data_vect;

    /* LOAD GRAPH */
    std::chrono::steady_clock::time_point startLoading = std::chrono::steady_clock::now();
    std::cout << "Loading starts at: " << getEpoch() << std::endl;

    COO_List* coo_list = load_graph_from_edge_list_file_to_coo(source_vertices_vect, destination_vertices_vect, edge_data_vect, argv[1]);

    // Convert the COO graph into a CSR format (for the in-memory GPU representation)
    CSC_List* csc_list = convert_coo_to_csc_format(coo_list->source, coo_list->destination, coo_list->edge_data);

    std::chrono::steady_clock::time_point endLoading = std::chrono::steady_clock::now();
    std::cout << "Loading ends at: " << getEpoch() << std::endl;

    /* EXECUTE SELECTED ALGORITHM & SAVE RESULTS */
    switch (SELECTED_ALGORITHM) {
        case "SSSP": {
            float* result = sssp(csc_list->source_indices, csc_list->destination_offsets,  csc_list->edge_data);
            save_sssp_result(result, argv[2]);
            break;
        }
        case "BFS": {
            int* result = bfs(csc_list->destination_offsets, csc_list->source_indices);
            save_bfs_result(result, argv[2]);
            break;
        }
        default: {
            std::cout << "Selected algorithm does not exist.";
        }
    }

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
 printf("\n printing COO\n"); 
   for (int i = 0 ; i<SIZE_EDGES; i++) {
        printf("\n(%d,%d) - %f", source[i], target[i], weight[i]);
    }
}

#include <stdio.h>
#include <cuda_runtime.h>
#include <Nvgraph.h>
#include <curand.h>
#include <curand_kernel.h>

void print_output(float *results, int nvertices);

void check(NvgraphStatus_t status) {
    if (status != Nvgraph_STATUS_SUCCESS) {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

// NVIDIA's SSSP implementation using Nvgraph: https://docs.nvidia.com/cuda/Nvgraph/index.html#Nvgraph-sssp-example
int main(int argc, char **argv) {
    const size_t  n = 6, nnz = 10, vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;

    // Nvgraph variables
    NvgraphStatus_t status; NvgraphHandle_t handle;
    NvgraphGraphDescr_t graph;
    NvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (NvgraphCSCTopology32I_t) malloc(sizeof(struct NvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
    float weights_h[] = {0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5};
    int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};

    check(NvgraphCreate(&handle));

    check(NvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    check(NvgraphSetGraphStructure(handle, graph, (void*)CSC_input, Nvgraph_CSC_32));
    check(NvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(NvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(NvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    // Solve
    int source_vert = 0;
    check(NvgraphSssp(handle, graph, 0,  &source_vert, 0));

    // Get and print result
    check(NvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));

    // Clean
    print_output(sssp_1_h, 6);
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(NvgraphDestroyGraphDescr(handle, graph));
    check(NvgraphDestroy(handle));

    return 0;
}

void print_output(float *results, int nvertices) {
    for (int i = 0; i < nvertices; i++) {
        printf("%f \n", results[i]);
    }
}
/*
 * Copyright 2015 Delft University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package science.atlarge.graphalytics.nvgraph.algorithms.bfs;

import science.atlarge.graphalytics.domain.algorithms.AlgorithmParameters;
import science.atlarge.graphalytics.domain.algorithms.BreadthFirstSearchParameters;
import science.atlarge.graphalytics.execution.RunSpecification;
import science.atlarge.graphalytics.domain.benchmark.BenchmarkRun;
import science.atlarge.graphalytics.nvgraph.NvgraphJob;
import science.atlarge.graphalytics.nvgraph.NvgraphConfiguration;

/**
 * Breadth-first Search job implementation for Nvgraph. This class is responsible for formatting BFS-specific
 * arguments to be passed to the platform executable, and does not include the implementation of the algorithm.
 *
 * @author Ahmed
 */
public final class BreadthFirstSearchJob extends NvgraphJob {

	private final long sourceVertex;

	/**
	 * Creates a new BreadthFirstSearchJob object with all mandatory parameters specified.
	 *
	 * @param platformConfig the platform configuration.
	 * @param inputPath the path to the input graph.
	 */
	public BreadthFirstSearchJob(RunSpecification runSpecification, NvgraphConfiguration platformConfig,
								 String inputPath, String outputPath) {
		super(runSpecification, platformConfig, inputPath, outputPath);

		AlgorithmParameters parameters = runSpecification.getBenchmarkRun().getAlgorithmParameters();
		this.sourceVertex = ((BreadthFirstSearchParameters)parameters).getSourceVertex();
	}

	@Override
	protected void appendAlgorithmParameters() {
		System.out.println("Note: Appending BFS params!");
		args.add(String.valueOf(this.sourceVertex));
	}
}

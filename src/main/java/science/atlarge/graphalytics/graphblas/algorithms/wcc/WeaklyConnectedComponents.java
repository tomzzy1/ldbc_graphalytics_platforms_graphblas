package science.atlarge.graphalytics.graphblas.algorithms.wcc;

import science.atlarge.graphalytics.domain.algorithms.BreadthFirstSearchParameters;
import science.atlarge.graphalytics.domain.graph.Graph;
import science.atlarge.graphalytics.execution.RunSpecification;
import science.atlarge.graphalytics.graphblas.GraphblasConfiguration;
import science.atlarge.graphalytics.graphblas.GraphblasJob;

/**
 * Breadth First Search job implementation for GraphBLAS. This class is responsible for formatting BFS-specific
 * arguments to be passed to the platform executable, and does not include the implementation of the algorithm.
 *
 * @author Bálint Hegyi
 */
public final class WeaklyConnectedComponents extends GraphblasJob {

	/**
	 * Creates a new BreadthFirstSearchJob object with all mandatory parameters specified.
	 *  @param platformConfig the platform configuration.
	 * @param inputDir the path to the input graph.
	 */
	public WeaklyConnectedComponents(RunSpecification runSpecification, GraphblasConfiguration platformConfig,
                                     String inputDir, String outputPath, Graph benchmarkGraph) {
		super(runSpecification, platformConfig, inputDir, outputPath, benchmarkGraph);
	}

	@Override
	protected void appendAlgorithmParameters() {
		commandLine.addArgument("--algorithm");
		commandLine.addArgument("wcc");
	}
}

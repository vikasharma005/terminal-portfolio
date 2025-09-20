import type { CommandComponentProps, ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';
import { useState } from 'preact/hooks';

const messages = i18n('challenge', {
	actualOutput: 'Your Output:',
	challengeCompleted: 'Challenge Completed! 🎉',
	checkSolution: 'Check Solution',
	correctSolution: 'Correct! Well done!',
	expectedOutput: 'Expected Output:',
	nextChallenge: 'Next Challenge',
	selectChallenge: 'Select a Challenge',
	startChallenge: 'Start Challenge',
	submitSolution: 'Submit Solution',
	title: 'Coding Challenges',
	wrongSolution: 'Not quite right. Try again!',
	yourSolution: 'Your Solution:',
});

const challenges = [
	// EASY CHALLENGES
	{
		description:
			'Implement a simple linear regression function to predict values. Given training data (x, y pairs), find the best fit line and predict new values.',
		difficulty: 'Easy',
		example: 'Input: x=[1,2,3,4], y=[2,4,6,8], predict_x=5\nOutput: 10 (y = 2x)',
		id: 1,
		solution: `function linearRegression(x, y, predictX) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return Math.round((slope * predictX + intercept) * 100) / 100;
}`,
		testCases: [
			{ expected: '10', input: '[1,2,3,4], [2,4,6,8], 5' },
			{ expected: '7', input: '[1,2,3], [3,5,7], 4' },
			{ expected: '0', input: '[0,1,2], [0,0,0], 3' },
		],
		title: '01. Linear Regression - Basic ML Algorithm',
	},
	{
		description:
			'Implement a simple k-means clustering algorithm. Given data points and k clusters, assign each point to the nearest centroid.',
		difficulty: 'Easy',
		example: 'Input: points=[[1,1],[2,2],[8,8],[9,9]], k=2\nOutput: [0,0,1,1] (cluster assignments)',
		id: 2,
		solution: `function kMeansClustering(points, k) {
    // Initialize centroids randomly
    let centroids = [];
    for (let i = 0; i < k; i++) {
        const randomIndex = Math.floor(Math.random() * points.length);
        centroids.push([...points[randomIndex]]);
    }
    
    let assignments = new Array(points.length).fill(0);
    let changed = true;
    
    while (changed) {
        changed = false;
        
        // Assign points to nearest centroid
        for (let i = 0; i < points.length; i++) {
            let minDist = Infinity;
            let newAssignment = 0;
            
            for (let j = 0; j < k; j++) {
                const dist = Math.sqrt(
                    Math.pow(points[i][0] - centroids[j][0], 2) + 
                    Math.pow(points[i][1] - centroids[j][1], 2)
                );
                if (dist < minDist) {
                    minDist = dist;
                    newAssignment = j;
                }
            }
            
            if (assignments[i] !== newAssignment) {
                assignments[i] = newAssignment;
                changed = true;
            }
        }
        
        // Update centroids
        for (let j = 0; j < k; j++) {
            const clusterPoints = points.filter((_, i) => assignments[i] === j);
            if (clusterPoints.length > 0) {
                centroids[j][0] = clusterPoints.reduce((sum, p) => sum + p[0], 0) / clusterPoints.length;
                centroids[j][1] = clusterPoints.reduce((sum, p) => sum + p[1], 0) / clusterPoints.length;
            }
        }
    }
    
    return assignments;
}`,
		testCases: [
			{ expected: '[0,0,1,1]', input: '[[1,1],[2,2],[8,8],[9,9]], 2' },
			{ expected: '[0,1,0,1]', input: '[[0,0],[10,10],[1,1],[9,9]], 2' },
			{ expected: '[0,0,0,0]', input: '[[1,1],[1,2],[2,1],[2,2]], 1' },
		],
		title: '02. K-Means Clustering - Unsupervised Learning',
	},

	// MEDIUM CHALLENGES
	{
		description:
			'Implement a Naive Bayes classifier for text classification. Given training data with features and labels, predict the class of new data.',
		difficulty: 'Medium',
		example:
			'Input: features=[[1,0,1],[0,1,1],[1,1,0]], labels=[0,1,0], new_features=[1,0,0]\nOutput: 0 (predicted class)',
		id: 3,
		solution: `function naiveBayesClassifier(features, labels, newFeatures) {
    const classes = [...new Set(labels)];
    const classCounts = {};
    const featureCounts = {};
    
    // Initialize counts
    classes.forEach(cls => {
        classCounts[cls] = 0;
        featureCounts[cls] = {};
    });
    
    // Count occurrences
    features.forEach((feature, i) => {
        const label = labels[i];
        classCounts[label]++;
        
        feature.forEach((val, j) => {
            if (!featureCounts[label][j]) featureCounts[label][j] = {};
            if (!featureCounts[label][j][val]) featureCounts[label][j][val] = 0;
            featureCounts[label][j][val]++;
        });
    });
    
    // Calculate probabilities and predict
    let maxProb = -Infinity;
    let prediction = classes[0];
    
    classes.forEach(cls => {
        let prob = Math.log(classCounts[cls] / features.length);
        
        newFeatures.forEach((val, j) => {
            const count = featureCounts[cls][j] && featureCounts[cls][j][val] ? featureCounts[cls][j][val] : 0;
            const total = classCounts[cls];
            prob += Math.log((count + 1) / (total + 2)); // Laplace smoothing
        });
        
        if (prob > maxProb) {
            maxProb = prob;
            prediction = cls;
        }
    });
    
    return prediction;
}`,
		testCases: [
			{ expected: '0', input: '[[1,0,1],[0,1,1],[1,1,0]], [0,1,0], [1,0,0]' },
			{ expected: '1', input: '[[1,0],[0,1],[1,1]], [0,1,1], [0,1]' },
			{ expected: '0', input: '[[1,1],[0,0]], [0,1], [1,0]' },
		],
		title: '03. Naive Bayes Classifier - Probabilistic ML',
	},
	{
		description:
			'Implement a decision tree algorithm for classification. Given training data, build a tree and make predictions on new data.',
		difficulty: 'Medium',
		example:
			'Input: features=[[1,0],[0,1],[1,1],[0,0]], labels=[1,1,0,0], new_features=[1,0]\nOutput: 1 (predicted class)',
		id: 4,
		solution: `function decisionTree(features, labels, newFeatures) {
    function calculateEntropy(labels) {
        const counts = {};
        labels.forEach(label => counts[label] = (counts[label] || 0) + 1);
        const total = labels.length;
        let entropy = 0;
        
        Object.values(counts).forEach(count => {
            const p = count / total;
            entropy -= p * Math.log2(p);
        });
        
        return entropy;
    }
    
    function findBestSplit(features, labels) {
        let bestGain = 0;
        let bestFeature = 0;
        let bestThreshold = 0;
        
        const parentEntropy = calculateEntropy(labels);
        
        for (let feature = 0; feature < features[0].length; feature++) {
            const values = features.map(f => f[feature]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
            
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                
                const leftIndices = features.map((f, idx) => f[feature] <= threshold ? idx : -1).filter(i => i !== -1);
                const rightIndices = features.map((f, idx) => f[feature] > threshold ? idx : -1).filter(i => i !== -1);
                
                const leftLabels = leftIndices.map(i => labels[i]);
                const rightLabels = rightIndices.map(i => labels[i]);
                
                const leftEntropy = calculateEntropy(leftLabels);
                const rightEntropy = calculateEntropy(rightLabels);
                
                const weightedEntropy = (leftLabels.length / labels.length) * leftEntropy + 
                                      (rightLabels.length / labels.length) * rightEntropy;
                
                const gain = parentEntropy - weightedEntropy;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }
        
        return { feature: bestFeature, threshold: bestThreshold };
    }
    
    // Simple decision tree with one split
    const { feature, threshold } = findBestSplit(features, labels);
    
    // Make prediction
    if (newFeatures[feature] <= threshold) {
        const leftIndices = features.map((f, idx) => f[feature] <= threshold ? idx : -1).filter(i => i !== -1);
        const leftLabels = leftIndices.map(i => labels[i]);
        const counts = {};
        leftLabels.forEach(label => counts[label] = (counts[label] || 0) + 1);
        return parseInt(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b));
    } else {
        const rightIndices = features.map((f, idx) => f[feature] > threshold ? idx : -1).filter(i => i !== -1);
        const rightLabels = rightIndices.map(i => labels[i]);
        const counts = {};
        rightLabels.forEach(label => counts[label] = (counts[label] || 0) + 1);
        return parseInt(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b));
    }
}`,
		testCases: [
			{ expected: '1', input: '[[1,0],[0,1],[1,1],[0,0]], [1,1,0,0], [1,0]' },
			{ expected: '0', input: '[[1,1],[0,0],[1,0],[0,1]], [0,0,1,1], [0,1]' },
			{ expected: '1', input: '[[2,1],[1,2],[2,2],[1,1]], [1,1,0,0], [2,1]' },
		],
		title: '04. Decision Tree - Tree-Based ML',
	},
	{
		description:
			'Implement a simple neural network with one hidden layer for binary classification. Use sigmoid activation and backpropagation.',
		difficulty: 'Medium',
		example:
			'Input: features=[[1,0],[0,1],[1,1],[0,0]], labels=[1,1,0,0], new_features=[1,0]\nOutput: 0.8 (prediction probability)',
		id: 5,
		solution: `function neuralNetwork(features, labels, newFeatures) {
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    function sigmoidDerivative(x) {
        return x * (1 - x);
    }
    
    // Initialize weights randomly
    const inputSize = features[0].length;
    const hiddenSize = 3;
    const outputSize = 1;
    
    let weights1 = Array(inputSize).fill().map(() => 
        Array(hiddenSize).fill().map(() => Math.random() * 2 - 1)
    );
    let weights2 = Array(hiddenSize).fill().map(() => 
        Array(outputSize).fill().map(() => Math.random() * 2 - 1)
    );
    
    const learningRate = 0.1;
    const epochs = 1000;
    
    // Training
    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = 0; i < features.length; i++) {
            const input = features[i];
            const target = labels[i];
            
            // Forward pass
            const hidden = Array(hiddenSize).fill(0);
            for (let j = 0; j < hiddenSize; j++) {
                for (let k = 0; k < inputSize; k++) {
                    hidden[j] += input[k] * weights1[k][j];
                }
                hidden[j] = sigmoid(hidden[j]);
            }
            
            const output = Array(outputSize).fill(0);
            for (let j = 0; j < outputSize; j++) {
                for (let k = 0; k < hiddenSize; k++) {
                    output[j] += hidden[k] * weights2[k][j];
                }
                output[j] = sigmoid(output[j]);
            }
            
            // Backward pass
            const outputError = target - output[0];
            const outputDelta = outputError * sigmoidDerivative(output[0]);
            
            const hiddenError = Array(hiddenSize).fill(0);
            const hiddenDelta = Array(hiddenSize).fill(0);
            
            for (let j = 0; j < hiddenSize; j++) {
                hiddenError[j] = outputDelta * weights2[j][0];
                hiddenDelta[j] = hiddenError[j] * sigmoidDerivative(hidden[j]);
            }
            
            // Update weights
            for (let j = 0; j < hiddenSize; j++) {
                weights2[j][0] += learningRate * outputDelta * hidden[j];
            }
            
            for (let j = 0; j < inputSize; j++) {
                for (let k = 0; k < hiddenSize; k++) {
                    weights1[j][k] += learningRate * hiddenDelta[k] * input[j];
                }
            }
        }
    }
    
    // Prediction
    const input = newFeatures;
    const hidden = Array(hiddenSize).fill(0);
    for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < inputSize; k++) {
            hidden[j] += input[k] * weights1[k][j];
        }
        hidden[j] = sigmoid(hidden[j]);
    }
    
    const output = Array(outputSize).fill(0);
    for (let j = 0; j < outputSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
            output[j] += hidden[k] * weights2[k][j];
        }
        output[j] = sigmoid(output[j]);
    }
    
    return Math.round(output[0] * 100) / 100;
}`,
		testCases: [
			{ expected: '0.8', input: '[[1,0],[0,1],[1,1],[0,0]], [1,1,0,0], [1,0]' },
			{ expected: '0.2', input: '[[1,1],[0,0],[1,0],[0,1]], [0,0,1,1], [0,1]' },
			{ expected: '0.5', input: '[[2,1],[1,2],[2,2],[1,1]], [1,1,0,0], [1,1]' },
		],
		title: '05. Neural Network - Deep Learning Basics',
	},
	{
		description:
			'Implement a Support Vector Machine (SVM) classifier using gradient descent. Classify data points into two classes.',
		difficulty: 'Medium',
		example:
			'Input: features=[[1,1],[2,2],[3,3],[4,4]], labels=[1,1,-1,-1], new_features=[2.5,2.5]\nOutput: 1 (predicted class)',
		id: 6,
		solution: `function supportVectorMachine(features, labels, newFeatures) {
    const learningRate = 0.01;
    const epochs = 1000;
    const lambda = 0.1; // regularization parameter
    
    // Initialize weights and bias
    let weights = Array(features[0].length).fill(0);
    let bias = 0;
    
    // Training
    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = 0; i < features.length; i++) {
            const x = features[i];
            const y = labels[i];
            
            // Calculate prediction
            let prediction = bias;
            for (let j = 0; j < x.length; j++) {
                prediction += weights[j] * x[j];
            }
            
            // Hinge loss gradient
            if (y * prediction < 1) {
                // Update weights
                for (let j = 0; j < weights.length; j++) {
                    weights[j] = weights[j] - learningRate * (lambda * weights[j] - y * x[j]);
                }
                bias = bias + learningRate * y;
            } else {
                // Only regularization
                for (let j = 0; j < weights.length; j++) {
                    weights[j] = weights[j] - learningRate * lambda * weights[j];
                }
            }
        }
    }
    
    // Prediction
    let prediction = bias;
    for (let j = 0; j < newFeatures.length; j++) {
        prediction += weights[j] * newFeatures[j];
    }
    
    return prediction > 0 ? 1 : -1;
}`,
		testCases: [
			{ expected: '1', input: '[[1,1],[2,2],[3,3],[4,4]], [1,1,-1,-1], [2.5,2.5]' },
			{ expected: '-1', input: '[[1,1],[2,2],[3,3],[4,4]], [1,1,-1,-1], [3.5,3.5]' },
			{ expected: '1', input: '[[0,0],[1,1],[2,2]], [1,1,-1], [0.5,0.5]' },
		],
		title: '06. Support Vector Machine - Margin-Based ML',
	},
	{
		description:
			'Implement a Random Forest classifier using multiple decision trees. Combine predictions from multiple trees for better accuracy.',
		difficulty: 'Medium',
		example:
			'Input: features=[[1,0],[0,1],[1,1],[0,0]], labels=[1,1,0,0], new_features=[1,0]\nOutput: 1 (predicted class)',
		id: 7,
		solution: `function randomForest(features, labels, newFeatures) {
    function createDecisionTree(features, labels) {
        function calculateEntropy(labels) {
            const counts = {};
            labels.forEach(label => counts[label] = (counts[label] || 0) + 1);
            const total = labels.length;
            let entropy = 0;
            
            Object.values(counts).forEach(count => {
                const p = count / total;
                entropy -= p * Math.log2(p);
            });
            
            return entropy;
        }
        
        function findBestSplit(features, labels) {
            let bestGain = 0;
            let bestFeature = 0;
            let bestThreshold = 0;
            
            const parentEntropy = calculateEntropy(labels);
            
            for (let feature = 0; feature < features[0].length; feature++) {
                const values = features.map(f => f[feature]);
                const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
                
                for (let i = 0; i < uniqueValues.length - 1; i++) {
                    const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                    
                    const leftIndices = features.map((f, idx) => f[feature] <= threshold ? idx : -1).filter(i => i !== -1);
                    const rightIndices = features.map((f, idx) => f[feature] > threshold ? idx : -1).filter(i => i !== -1);
                    
                    const leftLabels = leftIndices.map(i => labels[i]);
                    const rightLabels = rightIndices.map(i => labels[i]);
                    
                    const leftEntropy = calculateEntropy(leftLabels);
                    const rightEntropy = calculateEntropy(rightLabels);
                    
                    const weightedEntropy = (leftLabels.length / labels.length) * leftEntropy + 
                                          (rightLabels.length / labels.length) * rightEntropy;
                    
                    const gain = parentEntropy - weightedEntropy;
                    
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = threshold;
                    }
                }
            }
            
            return { feature: bestFeature, threshold: bestThreshold };
        }
        
        const { feature, threshold } = findBestSplit(features, labels);
        
        return { feature, threshold };
    }
    
    // Create multiple trees with bootstrap sampling
    const numTrees = 5;
    const trees = [];
    
    for (let i = 0; i < numTrees; i++) {
        // Bootstrap sample
        const sampleIndices = [];
        for (let j = 0; j < features.length; j++) {
            sampleIndices.push(Math.floor(Math.random() * features.length));
        }
        
        const sampleFeatures = sampleIndices.map(idx => features[idx]);
        const sampleLabels = sampleIndices.map(idx => labels[idx]);
        
        const tree = createDecisionTree(sampleFeatures, sampleLabels);
        trees.push(tree);
    }
    
    // Make predictions with all trees
    const predictions = trees.map(tree => {
        if (newFeatures[tree.feature] <= tree.threshold) {
            const leftIndices = features.map((f, idx) => f[tree.feature] <= tree.threshold ? idx : -1).filter(i => i !== -1);
            const leftLabels = leftIndices.map(i => labels[i]);
            const counts = {};
            leftLabels.forEach(label => counts[label] = (counts[label] || 0) + 1);
            return parseInt(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b));
        } else {
            const rightIndices = features.map((f, idx) => f[tree.feature] > tree.threshold ? idx : -1).filter(i => i !== -1);
            const rightLabels = rightIndices.map(i => labels[i]);
            const counts = {};
            rightLabels.forEach(label => counts[label] = (counts[label] || 0) + 1);
            return parseInt(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b));
        }
    });
    
    // Majority vote
    const counts = {};
    predictions.forEach(pred => counts[pred] = (counts[pred] || 0) + 1);
    return parseInt(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b));
}`,
		testCases: [
			{ expected: '1', input: '[[1,0],[0,1],[1,1],[0,0]], [1,1,0,0], [1,0]' },
			{ expected: '0', input: '[[1,1],[0,0],[1,0],[0,1]], [0,0,1,1], [0,1]' },
			{ expected: '1', input: '[[2,1],[1,2],[2,2],[1,1]], [1,1,0,0], [2,1]' },
		],
		title: '07. Random Forest - Ensemble Learning',
	},
	{
		description:
			'Implement a Gradient Boosting classifier using decision stumps. Build a strong classifier by combining weak learners.',
		difficulty: 'Medium',
		example:
			'Input: features=[[1,0],[0,1],[1,1],[0,0]], labels=[1,1,0,0], new_features=[1,0]\nOutput: 1 (predicted class)',
		id: 8,
		solution: `function gradientBoosting(features, labels, newFeatures) {
    const numIterations = 10;
    const learningRate = 0.1;
    
    // Initialize predictions with mean
    const meanLabel = labels.reduce((sum, label) => sum + label, 0) / labels.length;
    let predictions = Array(labels.length).fill(meanLabel);
    
    const weakLearners = [];
    
    for (let iteration = 0; iteration < numIterations; iteration++) {
        // Calculate residuals
        const residuals = labels.map((label, i) => label - predictions[i]);
        
        // Find best decision stump
        let bestFeature = 0;
        let bestThreshold = 0;
        let bestError = Infinity;
        
        for (let feature = 0; feature < features[0].length; feature++) {
            const values = features.map(f => f[feature]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
            
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                
                const leftIndices = features.map((f, idx) => f[feature] <= threshold ? idx : -1).filter(i => i !== -1);
                const rightIndices = features.map((f, idx) => f[feature] > threshold ? idx : -1).filter(i => i !== -1);
                
                const leftResiduals = leftIndices.map(i => residuals[i]);
                const rightResiduals = rightIndices.map(i => residuals[i]);
                
                const leftMean = leftResiduals.reduce((sum, r) => sum + r, 0) / leftResiduals.length;
                const rightMean = rightResiduals.reduce((sum, r) => sum + r, 0) / rightResiduals.length;
                
                let error = 0;
                leftIndices.forEach(i => error += Math.pow(residuals[i] - leftMean, 2));
                rightIndices.forEach(i => error += Math.pow(residuals[i] - rightMean, 2));
                
                if (error < bestError) {
                    bestError = error;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }
        
        // Calculate predictions for this stump
        const leftIndices = features.map((f, idx) => f[bestFeature] <= bestThreshold ? idx : -1).filter(i => i !== -1);
        const rightIndices = features.map((f, idx) => f[bestFeature] > bestThreshold ? idx : -1).filter(i => i !== -1);
        
        const leftResiduals = leftIndices.map(i => residuals[i]);
        const rightResiduals = rightIndices.map(i => residuals[i]);
        
        const leftMean = leftResiduals.reduce((sum, r) => sum + r, 0) / leftResiduals.length;
        const rightMean = rightResiduals.reduce((sum, r) => sum + r, 0) / rightResiduals.length;
        
        // Store weak learner
        weakLearners.push({
            feature: bestFeature,
            threshold: bestThreshold,
            leftValue: leftMean,
            rightValue: rightMean
        });
        
        // Update predictions
        for (let i = 0; i < features.length; i++) {
            if (features[i][bestFeature] <= bestThreshold) {
                predictions[i] += learningRate * leftMean;
            } else {
                predictions[i] += learningRate * rightMean;
            }
        }
    }
    
    // Make final prediction
    let finalPrediction = meanLabel;
    for (const learner of weakLearners) {
        if (newFeatures[learner.feature] <= learner.threshold) {
            finalPrediction += learningRate * learner.leftValue;
        } else {
            finalPrediction += learningRate * learner.rightValue;
        }
    }
    
    return finalPrediction > 0.5 ? 1 : 0;
}`,
		testCases: [
			{ expected: '1', input: '[[1,0],[0,1],[1,1],[0,0]], [1,1,0,0], [1,0]' },
			{ expected: '0', input: '[[1,1],[0,0],[1,0],[0,1]], [0,0,1,1], [0,1]' },
			{ expected: '1', input: '[[2,1],[1,2],[2,2],[1,1]], [1,1,0,0], [2,1]' },
		],
		title: '08. Gradient Boosting - Ensemble Learning',
	},
	{
		description:
			'Implement a Principal Component Analysis (PCA) algorithm to reduce dimensionality. Find the principal components and project data.',
		difficulty: 'Medium',
		example:
			'Input: data=[[1,2],[2,3],[3,4],[4,5]], components=1\nOutput: [[-1.41],[-0.71],[0.71],[1.41]] (projected data)',
		id: 9,
		solution: `function principalComponentAnalysis(data, numComponents) {
    const n = data.length;
    const m = data[0].length;
    
    // Center the data
    const mean = Array(m).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            mean[j] += data[i][j];
        }
    }
    for (let j = 0; j < m; j++) {
        mean[j] /= n;
    }
    
    const centeredData = data.map(row => 
        row.map((val, j) => val - mean[j])
    );
    
    // Calculate covariance matrix
    const covariance = Array(m).fill().map(() => Array(m).fill(0));
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += centeredData[k][i] * centeredData[k][j];
            }
            covariance[i][j] = sum / (n - 1);
        }
    }
    
    // Simple eigenvalue decomposition (for 2x2 matrix)
    if (m === 2) {
        const a = covariance[0][0];
        const b = covariance[0][1];
        const c = covariance[1][0];
        const d = covariance[1][1];
        
        const trace = a + d;
        const det = a * d - b * c;
        
        const discriminant = trace * trace - 4 * det;
        const lambda1 = (trace + Math.sqrt(discriminant)) / 2;
        const lambda2 = (trace - Math.sqrt(discriminant)) / 2;
        
        // Find eigenvector for largest eigenvalue
        let eigenvector;
        if (lambda1 > lambda2) {
            if (b !== 0) {
                eigenvector = [1, (lambda1 - a) / b];
            } else {
                eigenvector = [1, 0];
            }
        } else {
            if (b !== 0) {
                eigenvector = [1, (lambda2 - a) / b];
            } else {
                eigenvector = [1, 0];
            }
        }
        
        // Normalize eigenvector
        const norm = Math.sqrt(eigenvector[0] * eigenvector[0] + eigenvector[1] * eigenvector[1]);
        eigenvector[0] /= norm;
        eigenvector[1] /= norm;
        
        // Project data
        const projected = centeredData.map(row => [
            row[0] * eigenvector[0] + row[1] * eigenvector[1]
        ]);
        
        return projected.map(row => [Math.round(row[0] * 100) / 100]);
    }
    
    // For higher dimensions, return simplified result
    return centeredData.map(row => [Math.round(row[0] * 100) / 100]);
}`,
		testCases: [
			{ expected: '[[-1.41],[-0.71],[0.71],[1.41]]', input: '[[1,2],[2,3],[3,4],[4,5]], 1' },
			{ expected: '[[-0.71],[0.71]]', input: '[[1,1],[2,2]], 1' },
			{ expected: '[[0],[0]]', input: '[[0,0],[0,0]], 1' },
		],
		title: '09. Principal Component Analysis - Dimensionality Reduction',
	},

	// HARD CHALLENGES
	{
		description:
			'Implement a Convolutional Neural Network (CNN) for image classification. Build a simple CNN with convolution, pooling, and fully connected layers.',
		difficulty: 'Hard',
		example:
			'Input: image=[[1,0,1],[0,1,0],[1,0,1]], kernel=[[1,0],[0,1]], stride=1\nOutput: [[1,1],[1,1]] (feature map)',
		id: 10,
		solution: `function convolutionalNeuralNetwork(image, kernel, stride) {
    function convolution2D(image, kernel, stride) {
        const imageHeight = image.length;
        const imageWidth = image[0].length;
        const kernelHeight = kernel.length;
        const kernelWidth = kernel[0].length;
        
        const outputHeight = Math.floor((imageHeight - kernelHeight) / stride) + 1;
        const outputWidth = Math.floor((imageWidth - kernelWidth) / stride) + 1;
        
        const output = Array(outputHeight).fill().map(() => Array(outputWidth).fill(0));
        
        for (let i = 0; i < outputHeight; i++) {
            for (let j = 0; j < outputWidth; j++) {
                let sum = 0;
                for (let ki = 0; ki < kernelHeight; ki++) {
                    for (let kj = 0; kj < kernelWidth; kj++) {
                        const imageRow = i * stride + ki;
                        const imageCol = j * stride + kj;
                        sum += image[imageRow][imageCol] * kernel[ki][kj];
                    }
                }
                output[i][j] = sum;
            }
        }
        
        return output;
    }
    
    function maxPooling2D(featureMap, poolSize, stride) {
        const height = featureMap.length;
        const width = featureMap[0].length;
        const outputHeight = Math.floor((height - poolSize) / stride) + 1;
        const outputWidth = Math.floor((width - poolSize) / stride) + 1;
        
        const output = Array(outputHeight).fill().map(() => Array(outputWidth).fill(0));
        
        for (let i = 0; i < outputHeight; i++) {
            for (let j = 0; j < outputWidth; j++) {
                let maxVal = -Infinity;
                for (let pi = 0; pi < poolSize; pi++) {
                    for (let pj = 0; pj < poolSize; pj++) {
                        const row = i * stride + pi;
                        const col = j * stride + pj;
                        maxVal = Math.max(maxVal, featureMap[row][col]);
                    }
                }
                output[i][j] = maxVal;
            }
        }
        
        return output;
    }
    
    function relu(x) {
        return Math.max(0, x);
    }
    
    function softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((sum, x) => sum + x, 0);
        return expLogits.map(x => x / sumExp);
    }
    
    // Forward pass through CNN
    const conv1 = convolution2D(image, kernel, stride);
    const relu1 = conv1.map(row => row.map(val => relu(val)));
    const pool1 = maxPooling2D(relu1, 2, 2);
    
    // Flatten for fully connected layer
    const flattened = pool1.flat();
    
    // Simple fully connected layer (random weights for demo)
    const weights = Array(flattened.length).fill().map(() => Math.random() * 2 - 1);
    const bias = Math.random() * 2 - 1;
    
    let logits = bias;
    for (let i = 0; i < flattened.length; i++) {
        logits += flattened[i] * weights[i];
    }
    
    const probabilities = softmax([logits, -logits]);
    
    return {
        featureMap: conv1,
        pooled: pool1,
        prediction: probabilities[0] > probabilities[1] ? 1 : 0,
        confidence: Math.max(...probabilities)
    };
}`,
		testCases: [
			{
				expected: '{"featureMap":[[1,1],[1,1]],"pooled":[[1]],"prediction":1,"confidence":0.5}',
				input: '[[1,0,1],[0,1,0],[1,0,1]], [[1,0],[0,1]], 1',
			},
			{
				expected: '{"featureMap":[[0,0],[0,0]],"pooled":[[0]],"prediction":0,"confidence":0.5}',
				input: '[[0,0,0],[0,0,0],[0,0,0]], [[1,0],[0,1]], 1',
			},
			{
				expected: '{"featureMap":[[2,2],[2,2]],"pooled":[[2]],"prediction":1,"confidence":0.5}',
				input: '[[1,1,1],[1,1,1],[1,1,1]], [[1,0],[0,1]], 1',
			},
		],
		title: '10. Convolutional Neural Network - Deep Learning',
	},
	{
		description:
			'Implement a Recurrent Neural Network (RNN) with LSTM cells for sequence prediction. Build a simple LSTM for time series forecasting.',
		difficulty: 'Hard',
		example: 'Input: sequence=[1,2,3,4,5], hidden_size=2, steps=3\nOutput: [6,7,8] (predicted next values)',
		id: 11,
		solution: `function recurrentNeuralNetwork(sequence, hiddenSize, steps) {
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    function tanh(x) {
        return Math.tanh(x);
    }
    
    // Initialize LSTM parameters
    const inputSize = 1;
    const outputSize = 1;
    
    // Weights for LSTM gates
    const Wf = Array(hiddenSize).fill().map(() => Array(inputSize + hiddenSize).fill().map(() => Math.random() * 0.1));
    const Wi = Array(hiddenSize).fill().map(() => Array(inputSize + hiddenSize).fill().map(() => Math.random() * 0.1));
    const Wo = Array(hiddenSize).fill().map(() => Array(inputSize + hiddenSize).fill().map(() => Math.random() * 0.1));
    const Wc = Array(hiddenSize).fill().map(() => Array(inputSize + hiddenSize).fill().map(() => Math.random() * 0.1));
    
    // Output weights
    const Wy = Array(outputSize).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.1));
    
    // Biases
    const bf = Array(hiddenSize).fill(0);
    const bi = Array(hiddenSize).fill(0);
    const bo = Array(hiddenSize).fill(0);
    const bc = Array(hiddenSize).fill(0);
    const by = Array(outputSize).fill(0);
    
    // Initialize hidden state and cell state
    let h = Array(hiddenSize).fill(0);
    let c = Array(hiddenSize).fill(0);
    
    const predictions = [];
    
    // Process input sequence
    for (let t = 0; t < sequence.length; t++) {
        const input = [sequence[t]];
        const combined = [...input, ...h];
        
        // Forget gate
        const ft = Array(hiddenSize).fill(0);
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < combined.length; j++) {
                ft[i] += Wf[i][j] * combined[j];
            }
            ft[i] = sigmoid(ft[i] + bf[i]);
        }
        
        // Input gate
        const it = Array(hiddenSize).fill(0);
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < combined.length; j++) {
                it[i] += Wi[i][j] * combined[j];
            }
            it[i] = sigmoid(it[i] + bi[i]);
        }
        
        // Candidate values
        const ct_tilde = Array(hiddenSize).fill(0);
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < combined.length; j++) {
                ct_tilde[i] += Wc[i][j] * combined[j];
            }
            ct_tilde[i] = tanh(ct_tilde[i] + bc[i]);
        }
        
        // Update cell state
        for (let i = 0; i < hiddenSize; i++) {
            c[i] = ft[i] * c[i] + it[i] * ct_tilde[i];
        }
        
        // Output gate
        const ot = Array(hiddenSize).fill(0);
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < combined.length; j++) {
                ot[i] += Wo[i][j] * combined[j];
            }
            ot[i] = sigmoid(ot[i] + bo[i]);
        }
        
        // Update hidden state
        for (let i = 0; i < hiddenSize; i++) {
            h[i] = ot[i] * tanh(c[i]);
        }
    }
    
    // Generate predictions
    for (let step = 0; step < steps; step++) {
        // Use last prediction as input
        const input = step === 0 ? [sequence[sequence.length - 1]] : [predictions[step - 1]];
        const combined = [...input, ...h];
        
        // LSTM forward pass
        const ft = Array(hiddenSize).fill(0);
        const it = Array(hiddenSize).fill(0);
        const ct_tilde = Array(hiddenSize).fill(0);
        const ot = Array(hiddenSize).fill(0);
        
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < combined.length; j++) {
                ft[i] += Wf[i][j] * combined[j];
                it[i] += Wi[i][j] * combined[j];
                ct_tilde[i] += Wc[i][j] * combined[j];
                ot[i] += Wo[i][j] * combined[j];
            }
            ft[i] = sigmoid(ft[i] + bf[i]);
            it[i] = sigmoid(it[i] + bi[i]);
            ct_tilde[i] = tanh(ct_tilde[i] + bc[i]);
            ot[i] = sigmoid(ot[i] + bo[i]);
        }
        
        // Update states
        for (let i = 0; i < hiddenSize; i++) {
            c[i] = ft[i] * c[i] + it[i] * ct_tilde[i];
            h[i] = ot[i] * tanh(c[i]);
        }
        
        // Generate output
        let output = 0;
        for (let i = 0; i < outputSize; i++) {
            for (let j = 0; j < hiddenSize; j++) {
                output += Wy[i][j] * h[j];
            }
            output += by[i];
        }
        
        predictions.push(Math.round(output * 100) / 100);
    }
    
    return predictions;
}`,
		testCases: [
			{ expected: '[6,7,8]', input: '[1,2,3,4,5], 2, 3' },
			{ expected: '[3,4,5]', input: '[1,2], 2, 3' },
			{ expected: '[2,3,4]', input: '[1], 2, 3' },
		],
		title: '11. Recurrent Neural Network - LSTM Sequence Learning',
	},
	{
		description:
			'Implement a Transformer model with self-attention mechanism for natural language processing. Build a simple transformer for text classification.',
		difficulty: 'Hard',
		example:
			'Input: text="hello world", vocab_size=100, d_model=64, num_heads=4\nOutput: {"attention_weights":[[0.5,0.5]],"output":[[0.1,0.2,0.3]]}',
		id: 12,
		solution: `function transformerModel(text, vocabSize, dModel, numHeads) {
    function softmax(x) {
        const max = Math.max(...x);
        const exp = x.map(val => Math.exp(val - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(val => val / sum);
    }
    
    function multiHeadAttention(query, key, value, numHeads) {
        const seqLen = query.length;
        const dK = dModel / numHeads;
        
        // Initialize random weights for demo
        const Wq = Array(dModel).fill().map(() => Array(dK).fill().map(() => Math.random() * 0.1));
        const Wk = Array(dModel).fill().map(() => Array(dK).fill().map(() => Math.random() * 0.1));
        const Wv = Array(dModel).fill().map(() => Array(dK).fill().map(() => Math.random() * 0.1));
        
        // Linear transformations
        const Q = Array(seqLen).fill().map(() => Array(dK).fill(0));
        const K = Array(seqLen).fill().map(() => Array(dK).fill(0));
        const V = Array(seqLen).fill().map(() => Array(dK).fill(0));
        
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < dK; j++) {
                for (let k = 0; k < dModel; k++) {
                    Q[i][j] += query[i][k] * Wq[k][j];
                    K[i][j] += key[i][k] * Wk[k][j];
                    V[i][j] += value[i][k] * Wv[k][j];
                }
            }
        }
        
        // Scaled dot-product attention
        const attentionScores = Array(seqLen).fill().map(() => Array(seqLen).fill(0));
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                for (let k = 0; k < dK; k++) {
                    attentionScores[i][j] += Q[i][k] * K[j][k];
                }
                attentionScores[i][j] /= Math.sqrt(dK);
            }
        }
        
        // Apply softmax
        const attentionWeights = attentionScores.map(row => softmax(row));
        
        // Apply attention to values
        const output = Array(seqLen).fill().map(() => Array(dK).fill(0));
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                for (let k = 0; k < dK; k++) {
                    output[i][k] += attentionWeights[i][j] * V[j][k];
                }
            }
        }
        
        return { attentionWeights, output };
    }
    
    function layerNorm(x) {
        const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
        const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;
        const std = Math.sqrt(variance + 1e-8);
        
        return x.map(val => (val - mean) / std);
    }
    
    function feedForward(x) {
        const hiddenSize = dModel * 4;
        const W1 = Array(dModel).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.1));
        const W2 = Array(hiddenSize).fill().map(() => Array(dModel).fill().map(() => Math.random() * 0.1));
        
        // First linear layer with ReLU
        const hidden = Array(hiddenSize).fill(0);
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < dModel; j++) {
                hidden[i] += x[j] * W1[j][i];
            }
            hidden[i] = Math.max(0, hidden[i]); // ReLU
        }
        
        // Second linear layer
        const output = Array(dModel).fill(0);
        for (let i = 0; i < dModel; i++) {
            for (let j = 0; j < hiddenSize; j++) {
                output[i] += hidden[j] * W2[j][i];
            }
        }
        
        return output;
    }
    
    // Tokenize text (simple character-level tokenization)
    const tokens = text.split('').map(char => char.charCodeAt(0) % vocabSize);
    
    // Create embeddings (random for demo)
    const embeddings = tokens.map(token => 
        Array(dModel).fill().map(() => Math.random() * 0.1)
    );
    
    // Self-attention
    const { attentionWeights, output: attentionOutput } = multiHeadAttention(
        embeddings, embeddings, embeddings, numHeads
    );
    
    // Add & Norm
    const residual = attentionOutput.map((row, i) => 
        row.map((val, j) => val + embeddings[i][j])
    );
    
    const normalized = residual.map(row => layerNorm(row));
    
    // Feed Forward
    const ffOutput = normalized.map(row => feedForward(row));
    
    // Final Add & Norm
    const finalOutput = ffOutput.map((row, i) => 
        row.map((val, j) => val + normalized[i][j])
    );
    
    const finalNormalized = finalOutput.map(row => layerNorm(row));
    
    return {
        attentionWeights: attentionWeights.map(row => row.map(val => Math.round(val * 100) / 100)),
        output: finalNormalized.map(row => row.map(val => Math.round(val * 100) / 100))
    };
}`,
		testCases: [
			{
				expected: '{"attentionWeights":[[0.5,0.5]],"output":[[0.1,0.2,0.3]]}',
				input: '"hello world", 100, 64, 4',
			},
			{ expected: '{"attentionWeights":[[1]],"output":[[0.1,0.2,0.3]]}', input: '"a", 100, 64, 4' },
			{
				expected: '{"attentionWeights":[[0.33,0.33,0.33]],"output":[[0.1,0.2,0.3]]}',
				input: '"abc", 100, 64, 4',
			},
		],
		title: '12. Transformer Model - Self-Attention & NLP',
	},
];

const Challenge: FunctionalComponent<CommandComponentProps> = ({ args: _commandArguments = [] }) => {
	const t = useStore(messages);
	const [selectedChallenge, setSelectedChallenge] = useState<(typeof challenges)[0] | undefined>();
	const [userSolution, setUserSolution] = useState('');
	const [showSolution, setShowSolution] = useState(false);
	const [result, setResult] = useState<undefined | { correct: boolean; message: string }>();

	const startChallenge = (challenge: (typeof challenges)[0]) => {
		setSelectedChallenge(challenge);
		setUserSolution('');
		setShowSolution(false);
		setResult(undefined);
	};

	// Function to run code tests
	const runCodeTests = (challenge: (typeof challenges)[0], code: string) => {
		const results = {
			allPassed: false,
			failedTests: 0,
			feedback: '',
		};

		try {
			// Create a safe function execution context
			// eslint-disable-next-line no-new-func
			const userFunction = new Function(`return ${code}`)() as (...arguments_: unknown[]) => unknown;

			// Test each test case
			for (const testCase of challenge.testCases) {
				let actualOutput = '';

				try {
					// Execute the function with test input
					switch (challenge.id) {
						case 1: {
							// Linear Regression
							const [x, y, predictX] = JSON.parse(testCase.input) as [number[], number[], number];
							actualOutput = String(userFunction(x, y, predictX));
							break;
						}
						case 2: {
							// K-Means Clustering
							const [points, k] = JSON.parse(testCase.input) as [number[][], number];
							actualOutput = JSON.stringify(userFunction(points, k));
							break;
						}
						case 3: {
							// Naive Bayes
							const [features, labels, newFeatures] = JSON.parse(testCase.input) as [
								number[][],
								number[],
								number[],
							];
							actualOutput = String(userFunction(features, labels, newFeatures));
							break;
						}
						case 4: {
							// Decision Tree
							const [features, labels, newFeatures] = JSON.parse(testCase.input) as [
								number[][],
								number[],
								number[],
							];
							actualOutput = String(userFunction(features, labels, newFeatures));
							break;
						}
						case 5: {
							// Neural Network
							const [features, labels, newFeatures] = JSON.parse(testCase.input) as [
								number[][],
								number[],
								number[],
							];
							actualOutput = String(userFunction(features, labels, newFeatures));
							break;
						}
						case 6: {
							// Support Vector Machine
							const [features, labels, newFeatures] = JSON.parse(testCase.input) as [
								number[][],
								number[],
								number[],
							];
							actualOutput = String(userFunction(features, labels, newFeatures));
							break;
						}
						case 7: {
							// Random Forest
							const [features, labels, newFeatures] = JSON.parse(testCase.input) as [
								number[][],
								number[],
								number[],
							];
							actualOutput = String(userFunction(features, labels, newFeatures));
							break;
						}
						case 8: {
							// Gradient Boosting
							const [features, labels, newFeatures] = JSON.parse(testCase.input) as [
								number[][],
								number[],
								number[],
							];
							actualOutput = String(userFunction(features, labels, newFeatures));
							break;
						}
						case 9: {
							// Principal Component Analysis
							const [data, components] = JSON.parse(testCase.input) as [number[][], number];
							actualOutput = JSON.stringify(userFunction(data, components));
							break;
						}
						case 10: {
							// Convolutional Neural Network
							const [image, kernel, stride] = JSON.parse(testCase.input) as [
								number[][],
								number[][],
								number,
							];
							actualOutput = JSON.stringify(userFunction(image, kernel, stride));
							break;
						}
						case 11: {
							// Recurrent Neural Network
							const [sequence, hiddenSize, steps] = JSON.parse(testCase.input) as [
								number[],
								number,
								number,
							];
							actualOutput = JSON.stringify(userFunction(sequence, hiddenSize, steps));
							break;
						}
						case 12: {
							// Transformer Model
							const [text, vocabSize, dModel, numberHeads] = JSON.parse(testCase.input) as [
								string,
								number,
								number,
								number,
							];
							actualOutput = JSON.stringify(userFunction(text, vocabSize, dModel, numberHeads));
							break;
						}
						// No default
					}

					// Compare with expected output
					if (actualOutput !== testCase.expected) {
						results.failedTests++;
						results.feedback += `\nInput: ${testCase.input}\nExpected: ${testCase.expected}\nGot: ${actualOutput}`;
					}
				} catch (testError) {
					results.failedTests++;
					results.feedback += `\nInput: ${testCase.input}\nError: ${testError instanceof Error ? testError.message : 'Runtime error'}`;
				}
			}

			results.allPassed = results.failedTests === 0;
			if (results.allPassed) {
				results.feedback = 'All test cases passed! 🎉';
			}
		} catch (error) {
			results.failedTests = challenge.testCases.length;
			results.feedback = `Code execution error: ${error instanceof Error ? error.message : 'Invalid syntax'}`;
		}

		return results;
	};

	const checkSolution = () => {
		if (!selectedChallenge || !userSolution.trim()) {
			return;
		}

		try {
			// Create a safe execution environment
			const testResults = runCodeTests(selectedChallenge, userSolution);
			setResult({
				correct: testResults.allPassed,
				message: testResults.allPassed
					? t.correctSolution
					: `${testResults.failedTests.toString()} test(s) failed. ${testResults.feedback}`,
			});
		} catch (error) {
			setResult({
				correct: false,
				message: `Error: ${error instanceof Error ? error.message : 'Invalid code syntax'}`,
			});
		}
	};

	const nextChallenge = () => {
		if (!selectedChallenge) {
			return;
		}
		const currentIndex = challenges.findIndex(c => c.id === selectedChallenge.id);
		const nextIndex = (currentIndex + 1) % challenges.length;
		const nextChallenge = challenges[nextIndex];
		if (nextChallenge) {
			startChallenge(nextChallenge);
		}
	};

	if (selectedChallenge) {
		return (
			<div
				className='terminal-line-history'
				onClick={event => event.stopPropagation()}
				onKeyDown={event => event.stopPropagation()}>
				<div
					style={{
						alignItems: 'center',
						display: 'flex',
						justifyContent: 'space-between',
						marginBottom: '1.5rem',
					}}>
					<h3>{selectedChallenge.title}</h3>
					<div style={{ display: 'flex', gap: '0.5rem' }}>
						<button
							onClick={() => setShowSolution(!showSolution)}
							style={{
								background: 'var(--color-primary)',
								border: 'none',
								borderRadius: '4px',
								color: 'white',
								cursor: 'pointer',
								fontSize: '0.9rem',
								padding: '0.5rem 1rem',
							}}>
							{showSolution ? 'Hide Solution' : 'Show Solution'}
						</button>
						<button
							onClick={nextChallenge}
							style={{
								background: 'var(--color-bg-200)',
								border: '1px solid var(--color-border)',
								borderRadius: '4px',
								color: 'var(--color-text)',
								cursor: 'pointer',
								fontSize: '0.9rem',
								padding: '0.5rem 1rem',
							}}>
							{t.nextChallenge}
						</button>
					</div>
				</div>

				<div style={{ marginBottom: '1rem' }}>
					<span
						style={{
							background:
								selectedChallenge.difficulty === 'Easy'
									? '#4CAF50'
									: selectedChallenge.difficulty === 'Medium'
										? '#FF9800'
										: '#F44336',
							borderRadius: '12px',
							color: 'white',
							fontSize: '0.8rem',
							fontWeight: 'bold',
							padding: '0.25rem 0.5rem',
						}}>
						{selectedChallenge.difficulty}
					</span>
				</div>

				<div style={{ marginBottom: '1.5rem' }}>
					<p style={{ lineHeight: '1.6', marginBottom: '1rem' }}>{selectedChallenge.description}</p>
					<div
						style={{
							background: 'var(--color-bg-100)',
							border: '1px solid var(--color-border)',
							borderRadius: '6px',
							fontFamily: 'monospace',
							fontSize: '0.9rem',
							padding: '1rem',
							whiteSpace: 'pre-line',
						}}>
						{selectedChallenge.example}
					</div>
				</div>

				{showSolution ? (
					<div>
						<h4 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>Solution:</h4>
						<pre
							style={{
								background: 'var(--color-bg-100)',
								border: '1px solid var(--color-border)',
								borderRadius: '6px',
								fontSize: '0.9rem',
								lineHeight: '1.4',
								overflow: 'auto',
								padding: '1rem',
							}}>
							{selectedChallenge.solution}
						</pre>
					</div>
				) : (
					<div>
						<label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.5rem' }}>
							{t.yourSolution}:
						</label>
						<textarea
							onBlur={event => event.stopPropagation()}
							onClick={event => event.stopPropagation()}
							onFocus={event => event.stopPropagation()}
							onInput={event => setUserSolution((event.target as HTMLTextAreaElement).value)}
							onKeyDown={event => event.stopPropagation()}
							onKeyPress={event => event.stopPropagation()}
							onKeyUp={event => event.stopPropagation()}
							placeholder='Write your solution here...'
							style={{
								background: 'var(--color-bg-100)',
								border: '1px solid var(--color-border)',
								borderRadius: '4px',
								color: 'var(--color-text)',
								fontFamily: 'monospace',
								fontSize: '0.9rem',
								height: '200px',
								outline: 'none',
								padding: '1rem',
								resize: 'vertical',
								width: '100%',
							}}
							value={userSolution}
						/>
						<div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
							<button
								disabled={!userSolution.trim()}
								onClick={checkSolution}
								style={{
									background: userSolution.trim() ? 'var(--color-primary)' : 'var(--color-bg-200)',
									border: 'none',
									borderRadius: '4px',
									color: userSolution.trim() ? 'white' : 'var(--color-text-200)',
									cursor: userSolution.trim() ? 'pointer' : 'not-allowed',
									fontSize: '1rem',
									padding: '0.75rem 1.5rem',
								}}>
								{t.checkSolution}
							</button>
							<button
								disabled={!userSolution.trim()}
								onClick={() => {
									try {
										// eslint-disable-next-line no-new-func
										const userFunction = new Function(`return ${userSolution}`)() as (
											...arguments_: unknown[]
										) => unknown;
										let output: unknown;

										// Set test inputs based on challenge type
										switch (selectedChallenge.id) {
											case 1: {
												// Linear Regression
												output = userFunction([1, 2, 3, 4], [2, 4, 6, 8], 5);

												break;
											}
											case 2: {
												// K-Means Clustering
												output = userFunction(
													[
														[1, 1],
														[2, 2],
														[8, 8],
														[9, 9],
													],
													2
												);
												break;
											}
											case 3: {
												// Naive Bayes
												output = userFunction(
													[
														[1, 0, 1],
														[0, 1, 1],
														[1, 1, 0],
													],
													[0, 1, 0],
													[1, 0, 0]
												);
												break;
											}
											case 4: {
												// Decision Tree
												output = userFunction(
													[
														[1, 0],
														[0, 1],
														[1, 1],
														[0, 0],
													],
													[1, 1, 0, 0],
													[1, 0]
												);
												break;
											}
											case 5: {
												// Neural Network
												output = userFunction(
													[
														[1, 0],
														[0, 1],
														[1, 1],
														[0, 0],
													],
													[1, 1, 0, 0],
													[1, 0]
												);
												break;
											}
											case 6: {
												// Support Vector Machine
												output = userFunction(
													[
														[1, 1],
														[2, 2],
														[3, 3],
														[4, 4],
													],
													[1, 1, -1, -1],
													[2.5, 2.5]
												);
												break;
											}
											case 7: {
												// Random Forest
												output = userFunction(
													[
														[1, 0],
														[0, 1],
														[1, 1],
														[0, 0],
													],
													[1, 1, 0, 0],
													[1, 0]
												);
												break;
											}
											case 8: {
												// Gradient Boosting
												output = userFunction(
													[
														[1, 0],
														[0, 1],
														[1, 1],
														[0, 0],
													],
													[1, 1, 0, 0],
													[1, 0]
												);
												break;
											}
											case 9: {
												// Principal Component Analysis
												output = userFunction(
													[
														[1, 2],
														[2, 3],
														[3, 4],
														[4, 5],
													],
													1
												);
												break;
											}
											case 10: {
												// Convolutional Neural Network
												output = userFunction(
													[
														[1, 0, 1],
														[0, 1, 0],
														[1, 0, 1],
													],
													[
														[1, 0],
														[0, 1],
													],
													1
												);
												break;
											}
											case 11: {
												// Recurrent Neural Network
												output = userFunction([1, 2, 3, 4, 5], 2, 3);
												break;
											}
											case 12: {
												// Transformer Model
												output = userFunction('hello world', 100, 64, 4);
												break;
											}
											// No default
										}

										setResult({
											correct: false,
											message: `Test Output: ${JSON.stringify(output)}\n\nThis is just a preview. Use "Check Solution" for full testing.`,
										});
									} catch (error) {
										setResult({
											correct: false,
											message: `Error: ${error instanceof Error ? error.message : 'Invalid code syntax'}`,
										});
									}
								}}
								style={{
									background: userSolution.trim() ? '#FF9800' : 'var(--color-bg-200)',
									border: 'none',
									borderRadius: '4px',
									color: userSolution.trim() ? 'white' : 'var(--color-text-200)',
									cursor: userSolution.trim() ? 'pointer' : 'not-allowed',
									fontSize: '1rem',
									padding: '0.75rem 1.5rem',
								}}>
								🚀 Run Code
							</button>
						</div>

						{result && (
							<div
								style={{
									background: result.correct ? '#E8F5E8' : '#FFEBEE',
									border: `1px solid ${result.correct ? '#4CAF50' : '#F44336'}`,
									borderRadius: '4px',
									color: result.correct ? '#2E7D32' : '#C62828',
									marginTop: '1rem',
									padding: '1rem',
								}}>
								<div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>
									{result.correct ? '✅ Success!' : '❌ Test Failed'}
								</div>
								<div
									style={{
										background: 'rgba(0,0,0,0.05)',
										borderRadius: '4px',
										fontFamily: 'monospace',
										fontSize: '0.9rem',
										marginTop: '0.5rem',
										padding: '0.5rem',
										whiteSpace: 'pre-line',
									}}>
									{result.message}
								</div>
								{!result.correct && (
									<div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
										💡 <strong>Tip:</strong> Check your logic and make sure your function returns
										the expected output format.
									</div>
								)}
							</div>
						)}
					</div>
				)}
			</div>
		);
	}

	return (
		<div
			className='terminal-line-history'
			onClick={event => event.stopPropagation()}
			onKeyDown={event => event.stopPropagation()}>
			<h3>{t.title}</h3>
			<p style={{ color: 'var(--color-text-200)', marginBottom: '1.5rem' }}>
				Test your coding skills with these interactive challenges!
			</p>

			<div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
				{challenges.map(challenge => (
					<div
						key={challenge.id}
						onClick={() => startChallenge(challenge)}
						onMouseEnter={event => {
							event.currentTarget.style.transform = 'translateY(-2px)';
							event.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
						}}
						onMouseLeave={event => {
							event.currentTarget.style.transform = 'translateY(0)';
							event.currentTarget.style.boxShadow = 'none';
						}}
						style={{
							background: 'var(--color-bg-100)',
							border: '1px solid var(--color-border)',
							borderRadius: '8px',
							cursor: 'pointer',
							padding: '1.5rem',
							transition: 'all 0.3s ease',
						}}>
						<div
							style={{
								alignItems: 'center',
								display: 'flex',
								justifyContent: 'space-between',
								marginBottom: '1rem',
							}}>
							<h4 style={{ color: 'var(--color-text)', margin: 0 }}>{challenge.title}</h4>
							<span
								style={{
									background:
										challenge.difficulty === 'Easy'
											? '#4CAF50'
											: challenge.difficulty === 'Medium'
												? '#FF9800'
												: '#F44336',
									borderRadius: '12px',
									color: 'white',
									fontSize: '0.8rem',
									fontWeight: 'bold',
									padding: '0.25rem 0.5rem',
								}}>
								{challenge.difficulty}
							</span>
						</div>
						<p style={{ color: 'var(--color-text-200)', fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
							{challenge.description}
						</p>
						<div style={{ marginTop: '1rem', textAlign: 'center' }}>
							<span style={{ color: 'var(--color-primary)', fontWeight: 'bold' }}>
								{t.startChallenge} →
							</span>
						</div>
					</div>
				))}
			</div>
		</div>
	);
};

const ChallengeCommand: ComponentCommand = {
	command: 'challenge',
	component: Challenge,
};

export default ChallengeCommand;

#!/usr/bin/env python3
"""
Simple Example of using SynaptiCode

This example demonstrates how to use SynaptiCode to analyze a simple project
and propagate changes through the codebase.
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.analyzer import RelationshipAnalyzer
from src.core.propagator import ChangePropagator
from src.llm.cognitive import CognitiveProcessor

def main():
    """Run a simple example of SynaptiCode."""
    print("SynaptiCode Simple Example")
    print("-" * 30)
    
    # Step 1: Create a relationship analyzer
    print("\nStep 1: Creating a relationship analyzer")
    analyzer = RelationshipAnalyzer()
    
    # Step 2: Analyze a simple codebase (this would be an actual directory)
    print("\nStep 2: Analyzing a codebase")
    network = analyzer.build_network("./sample_codebase")
    
    # Step 3: Create a change propagator with the network
    print("\nStep 3: Creating a change propagator")
    propagator = ChangePropagator(network)
    
    # Step 4: Detect changes in a file
    print("\nStep 4: Detecting changes in a file")
    change = propagator.detect_change("./sample_codebase/main.py", "old content")
    
    # Step 5: Propagate changes through the network
    print("\nStep 5: Propagating changes")
    affected_components = propagator.propagate(change)
    
    # Step 6: Use cognitive processing for adaptation suggestions
    print("\nStep 6: Generating adaptation suggestions")
    cognitive = CognitiveProcessor()
    intent = cognitive.recognize_intent(change)
    adaptations = cognitive.generate_adaptations(affected_components, intent)
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()

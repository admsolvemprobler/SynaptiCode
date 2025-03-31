#!/usr/bin/env python3
"""
Command Line Interface for SynaptiCode

Provides a user-friendly interface for interacting with the SynaptiCode system.
"""

import argparse
import sys
import os

from src.core.analyzer import RelationshipAnalyzer
from src.core.propagator import ChangePropagator
from src.llm.cognitive import CognitiveProcessor

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SynaptiCode - The Living Codebase System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a codebase")
    analyze_parser.add_argument("path", help="Path to the codebase root")
    
    # Propagate command
    propagate_parser = subparsers.add_parser("propagate", help="Propagate changes in a codebase")
    propagate_parser.add_argument("file", help="Path to the changed file")
    propagate_parser.add_argument("--network", help="Path to a saved network file")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyzer = RelationshipAnalyzer()
        network = analyzer.build_network(args.path)
        print(f"Analysis complete. Network has {len(network)} nodes.")
    
    elif args.command == "propagate":
        if not args.network:
            print("Error: Network file is required for propagation.")
            sys.exit(1)
            
        # Here we would actually load the network, but for now just create a dummy
        network = {}
        propagator = ChangePropagator(network)
        
        # Here we would read the previous version, but for now just use a dummy
        previous_version = ""
        
        change = propagator.detect_change(args.file, previous_version)
        affected = propagator.propagate(change)
        
        print(f"Propagation complete. {len(affected)} components affected.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

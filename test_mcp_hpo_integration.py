"""
Test MCP HPO Integration

This script verifies that:
1. MCP Manager can connect to HPO server
2. HPO tools are loaded as LangChain tools
3. HPO tools return proper hyperparameter recommendations
"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plexe.langgraph.mcp_manager import MCPManager


async def test_mcp_hpo_integration():
    """Test MCP HPO server integration."""
    
    print("=" * 80)
    print("Testing MCP HPO Integration")
    print("=" * 80)
    
    # Initialize MCP Manager
    print("\n1. Initializing MCP Manager...")
    manager = MCPManager("mcp_config.json")
    
    try:
        await manager.initialize()
        print(f"✓ MCP Manager initialized successfully")
        print(f"  Connected to {len(manager.sessions)} MCP servers")
        
    except Exception as e:
        print(f"✗ Failed to initialize MCP Manager: {e}")
        return
    
    # Check loaded tools
    print("\n2. Checking loaded MCP tools...")
    tools = manager.get_tools()
    print(f"✓ Loaded {len(tools)} total MCP tools")
    
    # Find HPO-related tools
    hpo_tools = [t for t in tools if any(keyword in t.name.lower() 
                                         for keyword in ['hyperparameter', 'hpo', 'benchmark', 'compare'])]
    
    if not hpo_tools:
        print("✗ No HPO tools found!")
        print("\nAvailable tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:80]}...")
        return
    
    print(f"✓ Found {len(hpo_tools)} HPO-related tools:")
    for tool in hpo_tools:
        print(f"  - {tool.name}")
    
    # Test each HPO tool
    print("\n3. Testing HPO tools...")
    
    # Test search_optimal_hyperparameters
    print("\n3a. Testing search_optimal_hyperparameters...")
    search_tool = next((t for t in tools if 'search_optimal_hyperparameters' in t.name), None)
    if search_tool:
        try:
            result = search_tool.func(
                task_type="binary_classification",
                num_nodes=15000,
                num_tables=7,
                is_temporal=True,
                model_architecture="gnn"
            )
            print(f"✓ search_optimal_hyperparameters executed successfully")
            print(f"  Result type: {type(result)}")
            
            # Try to parse result
            import json
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if "hyperparameters" in result_dict:
                        print(f"  Hyperparameters found:")
                        for key, value in result_dict["hyperparameters"].items():
                            print(f"    - {key}: {value}")
                    if "reasoning" in result_dict:
                        print(f"  Reasoning: {result_dict['reasoning'][:100]}...")
                except:
                    print(f"  Raw result: {result[:200]}...")
            else:
                print(f"  Result: {result}")
                
        except Exception as e:
            print(f"✗ Error testing search_optimal_hyperparameters: {e}")
    else:
        print("✗ search_optimal_hyperparameters tool not found")
    
    # Test extract_hyperparameters_from_papers
    print("\n3b. Testing extract_hyperparameters_from_papers...")
    extract_tool = next((t for t in tools if 'extract_hyperparameters_from_papers' in t.name), None)
    if extract_tool:
        try:
            result = extract_tool.func(
                paper_query="Graph Neural Networks node classification",
                model_type="gnn",
                num_papers=3
            )
            print(f"✓ extract_hyperparameters_from_papers executed successfully")
            print(f"  Result type: {type(result)}")
            
            import json
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    print(f"  Papers analyzed: {result_dict.get('papers_analyzed', 'N/A')}")
                    print(f"  Papers with hyperparams: {result_dict.get('papers_with_hyperparams', 'N/A')}")
                    if result_dict.get('aggregated_hyperparameters'):
                        print(f"  Aggregated hyperparameters:")
                        for key, value in result_dict['aggregated_hyperparameters'].items():
                            print(f"    - {key}: {value}")
                except:
                    print(f"  Raw result: {result[:200]}...")
            else:
                print(f"  Result: {result}")
                
        except Exception as e:
            print(f"✗ Error testing extract_hyperparameters_from_papers: {e}")
    else:
        print("✗ extract_hyperparameters_from_papers tool not found")
    
    # Test get_benchmark_hyperparameters
    print("\n3c. Testing get_benchmark_hyperparameters...")
    benchmark_tool = next((t for t in tools if 'get_benchmark_hyperparameters' in t.name), None)
    if benchmark_tool:
        try:
            result = benchmark_tool.func(
                task_type="binary_classification",
                dataset_domain="relational",
                model_architecture="gnn"
            )
            print(f"✓ get_benchmark_hyperparameters executed successfully")
            print(f"  Result type: {type(result)}")
            
            import json
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if "hyperparameters" in result_dict:
                        print(f"  Hyperparameters found:")
                        for key, value in result_dict["hyperparameters"].items():
                            print(f"    - {key}: {value}")
                except:
                    print(f"  Raw result: {result[:200]}...")
            else:
                print(f"  Result: {result}")
                
        except Exception as e:
            print(f"✗ Error testing get_benchmark_hyperparameters: {e}")
    else:
        print("✗ get_benchmark_hyperparameters tool not found")
    
    # Test compare_hyperparameter_configs
    print("\n3d. Testing compare_hyperparameter_configs...")
    compare_tool = next((t for t in tools if 'compare_hyperparameter_configs' in t.name), None)
    if compare_tool:
        try:
            # Create mock configs
            import json
            configs = [
                {"hyperparameters": {"learning_rate": 0.01, "batch_size": 512, "hidden_channels": 128}, "source": "heuristic"},
                {"hyperparameters": {"learning_rate": 0.008, "batch_size": 512, "hidden_channels": 128}, "source": "literature"},
                {"hyperparameters": {"learning_rate": 0.01, "batch_size": 512, "hidden_channels": 256}, "source": "benchmark"}
            ]
            
            result = compare_tool.func(
                configs=configs,
                strategy="ensemble_median"
            )
            print(f"✓ compare_hyperparameter_configs executed successfully")
            print(f"  Result type: {type(result)}")
            
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if "recommended_hyperparameters" in result_dict:
                        print(f"  Recommended hyperparameters:")
                        for key, value in result_dict["recommended_hyperparameters"].items():
                            print(f"    - {key}: {value}")
                except:
                    print(f"  Raw result: {result[:200]}...")
            else:
                print(f"  Result: {result}")
                
        except Exception as e:
            print(f"✗ Error testing compare_hyperparameter_configs: {e}")
    else:
        print("✗ compare_hyperparameter_configs tool not found")
    
    print("\n" + "=" * 80)
    print("MCP HPO Integration Test Complete")
    print("=" * 80)
    
    # Clean up MCP connections properly
    print("\nCleaning up MCP connections...")
    try:
        await manager.close()
        print("✓ Cleanup complete")
    except Exception as e:
        print(f"Warning during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(test_mcp_hpo_integration())

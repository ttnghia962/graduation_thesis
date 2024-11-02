import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class FlowchartGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        
    def add_node_from_detection(self, id: int, text: str, shape: str, 
                              position: Tuple[int, int], confidence: float):
        """Add a node with detected properties"""
        self.G.add_node(id, text=text, shape=shape, 
                       pos=position, confidence=confidence)
    
    def build_relationships(self):
        """Build edges based on vertical positions"""
        nodes = list(self.G.nodes(data=True))
        
        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i+1:]:
                if self._is_connected(data1, data2):
                    weight = (data1['confidence'] + data2['confidence']) / 2
                    self.G.add_edge(node1, node2, weight=weight)
    
    def _is_connected(self, data1: Dict, data2: Dict) -> bool:
        """Check if two nodes should be connected"""
        x1, y1 = data1['pos']
        x2, y2 = data2['pos']
        
        # Basic vertical alignment check
        return (y2 > y1 and abs(x2 - x1) < 100)
    
    def analyze_structure(self) -> Dict:
        """Analyze flowchart structure"""
        analysis = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'root_nodes': [n for n in self.G.nodes() if self.G.in_degree(n) == 0],
            'leaf_nodes': [n for n in self.G.nodes() if self.G.out_degree(n) == 0],
            'cycles': list(nx.simple_cycles(self.G)),
            'levels': list(nx.topological_generations(self.G))
        }
        return analysis
    
    def find_paths(self, start_node: int, end_node: int) -> List[List[int]]:
        """Find all possible paths between two nodes"""
        return list(nx.all_simple_paths(self.G, start_node, end_node))
    
    def get_hierarchy_levels(self) -> Dict[int, int]:
        """Get hierarchy level for each node"""
        levels = {}
        for level_num, level_nodes in enumerate(nx.topological_generations(self.G)):
            for node in level_nodes:
                levels[node] = level_num + 1
        return levels
    
    def get_node_relationships(self) -> Dict[int, List[int]]:
        """Get parent-child relationships"""
        relationships = {}
        for node in self.G.nodes():
            relationships[node] = list(self.G.successors(node))
        return relationships
    
    def validate_flowchart(self) -> List[str]:
        """Validate flowchart structure"""
        issues = []
        
        # Check for cycles (invalid in most flowcharts)
        if len(list(nx.simple_cycles(self.G))) > 0:
            issues.append("Found cycles in flowchart")
        
        # Check for multiple root nodes
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        if len(roots) > 1:
            issues.append("Multiple start points detected")
        
        # Check for disconnected components
        if not nx.is_weakly_connected(self.G):
            issues.append("Flowchart has disconnected components")
        
        return issues
    
    def get_branching_points(self) -> List[int]:
        """Find decision points (nodes with multiple outgoing edges)"""
        return [n for n in self.G.nodes() if self.G.out_degree(n) > 1]
    
    def get_merge_points(self) -> List[int]:
        """Find merge points (nodes with multiple incoming edges)"""
        return [n for n in self.G.nodes() if self.G.in_degree(n) > 1]
    
    def export_to_mermaid(self) -> str:
        """Export flowchart to Mermaid format"""
        mermaid = ["graph TD"]
        
        for edge in self.G.edges():
            node1, node2 = edge
            text1 = self.G.nodes[node1]['text']
            text2 = self.G.nodes[node2]['text']
            mermaid.append(f"    {node1}[{text1}] --> {node2}[{text2}]")
            
        return "\n".join(mermaid)
    
    def visualize(self, layout_type: str = 'hierarchical'):
        """Visualize the flowchart"""
        if layout_type == 'hierarchical':
            pos = nx.spring_layout(self.G)
        elif layout_type == 'circular':
            pos = nx.circular_layout(self.G)
        else:
            pos = nx.kamada_kawai_layout(self.G)
            
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, 
                             node_color='lightblue', 
                             node_size=2000)
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, 
                             edge_color='gray', 
                             arrows=True)
        
        # Add labels
        labels = nx.get_node_attributes(self.G, 'text')
        nx.draw_networkx_labels(self.G, pos, labels)
        
        plt.title("Flowchart Visualization")
        plt.axis('off')
        plt.show()

def main():
    # Create example flowchart
    flow = FlowchartGraph()
    
    # Add nodes (normally these would come from image detection)
    flow.add_node_from_detection(1, "Start", "oval", (0, 0), 1.0)
    flow.add_node_from_detection(2, "Check Input", "diamond", (0, 1), 0.9)
    flow.add_node_from_detection(3, "Process Data", "rectangle", (-1, 2), 0.8)
    flow.add_node_from_detection(4, "Error Handler", "rectangle", (1, 2), 0.8)
    flow.add_node_from_detection(5, "End", "oval", (0, 3), 1.0)
    
    # Build relationships
    flow.build_relationships()
    
    # Analyze structure
    analysis = flow.analyze_structure()
    print("\nStructure Analysis:")
    print(f"Number of nodes: {analysis['num_nodes']}")
    print(f"Number of edges: {analysis['num_edges']}")
    print(f"Root nodes: {analysis['root_nodes']}")
    print(f"Leaf nodes: {analysis['leaf_nodes']}")
    
    # Get hierarchy levels
    levels = flow.get_hierarchy_levels()
    print("\nHierarchy Levels:")
    for node, level in levels.items():
        print(f"Node {node}: Level {level}")
    
    # Validate flowchart
    issues = flow.validate_flowchart()
    if issues:
        print("\nValidation Issues:")
        for issue in issues:
            print(f"- {issue}")
    
    # Export to Mermaid
    print("\nMermaid Format:")
    print(flow.export_to_mermaid())
    
    # Visualize
    flow.visualize()

if __name__ == "__main__":
    main()
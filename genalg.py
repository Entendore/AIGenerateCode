import random
import string
import numpy as np
from typing import List, Tuple, Callable, Dict, Any
import ast
from chromadb import Client, Documents
from chromadb.config import Settings
import textwrap
import traceback
import sys
from io import StringIO
import contextlib
import timeout_decorator
import unittest
from typing import Optional

class CodeTestResult:
    def __init__(self, success: bool, output: str, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error

class SafeCodeExecutor:
    """Safely executes code with timeout and sandbox restrictions."""
    
    @timeout_decorator.timeout(5, use_signals=False)
    def _run_with_timeout(self, code: str, input_data: Dict[str, Any]) -> CodeTestResult:
        """Run code with timeout."""
        # Create a dictionary for local variables
        local_vars = input_data.copy()
        
        # Redirect stdout to capture output
        output = StringIO()
        
        try:
            with contextlib.redirect_stdout(output):
                # First, try to compile the code to catch syntax errors
                compiled_code = compile(code, '<string>', 'exec')
                
                # Execute the code with restricted built-ins
                restricted_globals = {
                    '__builtins__': {
                        name: __builtins__[name]
                        for name in [
                            'range', 'len', 'list', 'dict', 'set', 'int',
                            'str', 'float', 'bool', 'tuple', 'map', 'filter',
                            'sum', 'min', 'max', 'abs', 'round', 'zip',
                            'print', 'TypeError', 'ValueError', 'Exception'
                        ]
                    }
                }
                
                exec(compiled_code, restricted_globals, local_vars)
                
            return CodeTestResult(
                success=True,
                output=output.getvalue(),
                error=None
            )
            
        except Exception as e:
            return CodeTestResult(
                success=False,
                output=output.getvalue(),
                error=str(e)
            )

    def run_code(self, code: str, input_data: Dict[str, Any] = None) -> CodeTestResult:
        """Main method to safely execute code."""
        if input_data is None:
            input_data = {}
            
        try:
            return self._run_with_timeout(code, input_data)
        except timeout_decorator.TimeoutError:
            return CodeTestResult(
                success=False,
                output="",
                error="Code execution timed out"
            )

class TestCase:
    def __init__(self, input_data: Dict[str, Any], expected_output: Any, test_name: str):
        self.input_data = input_data
        self.expected_output = expected_output
        self.test_name = test_name

class CodeGenome:
    def __init__(self, code: str = None, max_length: int = 200):
        self.max_length = max_length
        self.code = code if code else self._generate_random_code()
        
    def _generate_random_code(self) -> str:
        """Generate random Python-like code with function structure."""
        templates = [
            """
            def solution({}):
                result = 0
                {}
                return result
            """,
            """
            def solution({}):
                output = []
                {}
                return output
            """,
            """
            def solution({}):
                if {}:
                    return {}
                return {}
            """
        ]
        
        # Generate random parameter names
        num_params = random.randint(1, 3)
        params = [random.choice(string.ascii_lowercase) for _ in range(num_params)]
        
        # Generate random code body
        operations = [
            "result += " + random.choice(params),
            "result *= " + random.choice(params),
            f"for {random.choice(string.ascii_lowercase)} in {random.choice(params)}:",
            "output.append(" + random.choice(params) + ")",
            "if " + random.choice(params) + " > 0:",
        ]
        
        body = "\n    ".join(random.choice(operations) for _ in range(random.randint(2, 5)))
        template = random.choice(templates)
        
        return template.format(
            ", ".join(params),
            body,
            random.choice(params),
            random.choice(params)
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> None:
        """Mutate the code string with given probability."""
        if random.random() < mutation_rate:
            mutation_type = random.choice(['insert', 'delete', 'replace', 'swap'])
            
            lines = self.code.split('\n')
            if mutation_type == 'insert':
                # Insert a new line of code
                new_line = random.choice([
                    "    result += 1",
                    "    output.append(result)",
                    "    if result > 0: result *= 2",
                ])
                pos = random.randint(1, len(lines))
                lines.insert(pos, new_line)
            elif mutation_type == 'delete' and len(lines) > 3:
                # Delete a line while preserving function structure
                pos = random.randint(1, len(lines) - 2)
                lines.pop(pos)
            elif mutation_type == 'replace':
                # Replace a line with a new one
                pos = random.randint(1, len(lines) - 2)
                lines[pos] = "    " + random.choice([
                    "result = max(result, 0)",
                    "result = min(result, 100)",
                    "output.extend([result])",
                ])
            elif mutation_type == 'swap' and len(lines) > 3:
                # Swap two lines while preserving function structure
                pos1, pos2 = random.sample(range(1, len(lines) - 1), 2)
                lines[pos1], lines[pos2] = lines[pos2], lines[pos1]
                
            self.code = '\n'.join(lines)

class GeneticCodeGenerator:
    def __init__(self, 
                 population_size: int = 100,
                 test_cases: List[TestCase] = None,
                 collection_name: str = "generated_code"):
        self.population_size = population_size
        self.test_cases = test_cases or []
        self.population: List[CodeGenome] = []
        self.executor = SafeCodeExecutor()
        
        # Initialize ChromaDB
        self.chroma_client = Client(Settings(persist_directory="./code_db"))
        self.collection = self.chroma_client.create_collection(name=collection_name)
        
    def add_test_case(self, input_data: Dict[str, Any], expected_output: Any, test_name: str):
        """Add a test case to the fitness evaluation."""
        self.test_cases.append(TestCase(input_data, expected_output, test_name))
    
    def evaluate_fitness(self, code: str) -> Tuple[float, List[str]]:
        """Evaluate code fitness based on test cases and code quality."""
        if not self.test_cases:
            raise ValueError("No test cases defined")
            
        fitness = 0.0
        test_results = []
        
        # Check if code is valid Python
        try:
            ast.parse(code)
            fitness += 0.2  # 20% for valid syntax
        except SyntaxError:
            return 0.0, ["Syntax Error"]
        
        # Run test cases
        for test_case in self.test_cases:
            result = self.executor.run_code(code, test_case.input_data)
            
            if result.success:
                try:
                    # Get the solution function's output
                    local_vars = {}
                    exec(code, {}, local_vars)
                    solution_func = local_vars.get('solution')
                    if solution_func:
                        actual_output = solution_func(**test_case.input_data)
                        if actual_output == test_case.expected_output:
                            fitness += 0.8 / len(self.test_cases)  # Distribute remaining 80% across test cases
                            test_results.append(f"✓ {test_case.test_name}")
                        else:
                            test_results.append(
                                f"✗ {test_case.test_name}: Expected {test_case.expected_output}, got {actual_output}"
                            )
                    else:
                        test_results.append("✗ No solution function found")
                except Exception as e:
                    test_results.append(f"✗ Runtime error in {test_case.test_name}: {str(e)}")
            else:
                test_results.append(f"✗ {test_case.test_name}: {result.error}")
        
        return fitness, test_results
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = [CodeGenome() for _ in range(self.population_size)]
    
    def evaluate_population(self) -> List[Tuple[CodeGenome, float, List[str]]]:
        """Evaluate fitness for all individuals."""
        evaluated = []
        for genome in self.population:
            fitness, test_results = self.evaluate_fitness(genome.code)
            evaluated.append((genome, fitness, test_results))
        return sorted(evaluated, key=lambda x: x[1], reverse=True)
    
    def select_parents(self, evaluated_population: List[Tuple[CodeGenome, float, List[str]]]) -> List[CodeGenome]:
        """Select parents using tournament selection."""
        tournament_size = 3
        parents = []
        
        for _ in range(self.population_size):
            tournament = random.sample(evaluated_population, tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
            
        return parents
    
    def crossover(self, parent1: CodeGenome, parent2: CodeGenome) -> CodeGenome:
        """Perform crossover between two parents."""
        # Split both parents' code into lines
        parent1_lines = parent1.code.split('\n')
        parent2_lines = parent2.code.split('\n')
        
        # Keep function definition from parent1
        child_lines = [parent1_lines[0]]
        
        # Crossover the function body
        p1_body = parent1_lines[1:-1]
        p2_body = parent2_lines[1:-1]
        
        if p1_body and p2_body:
            crossover_point = random.randint(0, min(len(p1_body), len(p2_body)))
            child_body = p1_body[:crossover_point] + p2_body[crossover_point:]
            child_lines.extend(child_body)
        
        # Keep return statement from parent1
        child_lines.append(parent1_lines[-1])
        
        return CodeGenome('\n'.join(child_lines))
    
    def store_in_vector_db(self, code: str, fitness: float, test_results: List[str], metadata: dict = None) -> None:
        """
        Store generated code in the vector database.

        Args:
            code (str): The generated code.
            fitness (float): The fitness score of the code.
            test_results (List[str]): The test results of the code.
            metadata (dict, optional): Additional metadata to store with the code. Defaults to None.
        """
        document = code
        metadata_dict = {
            "fitness": fitness,
            "test_results": ', '.join(test_results),
            **(metadata or {})
        }

        self.collection.add(documents=[document], metadatas=[metadata_dict], ids=[f"code_{random.randint(0, 1000000)}"])

    
    def evolve(self, generations: int = 100, target_fitness: float = 0.95):
        """Main evolution loop."""
        self.initialize_population()
        
        for generation in range(generations):
            # Evaluate current population
            evaluated_population = self.evaluate_population()
            best_genome, best_fitness, best_results = evaluated_population[0]
            
            # Store best individual in vector DB
            self.store_in_vector_db(
                best_genome.code,
                best_fitness,
                best_results,
                {"generation": generation}
            )
            
            # Check if target fitness reached
            if best_fitness >= target_fitness:
                print(f"Target fitness reached at generation {generation}")
                print("\nBest solution:")
                print(best_genome.code)
                print("\nTest results:")
                for result in best_results:
                    print(result)
                break
            
            # Selection
            parents = self.select_parents(evaluated_population)
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child.mutate()
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 10 == 0:
                print(f"\nGeneration {generation}:")
                print(f"Best Fitness: {best_fitness}")
                print("Best Code:")
                print(best_genome.code)
                print("\nTest Results:")
                for result in best_results:
                    print(result)
'''
# Example usage
def main():
    # Create test cases
    test_cases = [
        TestCase(
            input_data={"numbers": [1, 2, 3, 4, 5]},
            expected_output=15,
            test_name="Sum of numbers"
        ),
        TestCase(
            input_data={"numbers": [-1, -2, 3, 4]},
            expected_output=4,
            test_name="Sum of positive numbers"
        ),
        TestCase(
            input_data={"numbers": []},
            expected_output=0,
            test_name="Empty list"
        )
    ]
    
    generator = GeneticCodeGenerator(
        population_size=50,
        test_cases=test_cases
    )
    
    generator.evolve(
        generations=100,
        target_fitness=0.9
    )

if __name__ == "__main__":
    main()
'''

# Example solving a coding problem: Create a function that processes a list of numbers
# Requirements:
# 1. Sum all positive numbers
# 2. Handle empty lists
# 3. Skip negative numbers
# 4. Return 0 if all numbers are negative

def main():
    # Define comprehensive test cases
    test_cases = [
        TestCase(
            input_data={"numbers": [1, -2, 3, -4, 5]},
            expected_output=9,  # 1 + 3 + 5
            test_name="Mix of positive and negative numbers"
        ),
        TestCase(
            input_data={"numbers": []},
            expected_output=0,
            test_name="Empty list"
        ),
        TestCase(
            input_data={"numbers": [-1, -2, -3]},
            expected_output=0,
            test_name="All negative numbers"
        ),
        TestCase(
            input_data={"numbers": [1, 2, 3]},
            expected_output=6,
            test_name="All positive numbers"
        ),
        TestCase(
            input_data={"numbers": [0, 0, 0]},
            expected_output=0,
            test_name="All zeros"
        ),
        TestCase(
            input_data={"numbers": [1000, 2000, 3000]},
            expected_output=6000,
            test_name="Large numbers"
        )
    ]
    
    # Initialize the generator with more generations for complex problems
    generator = GeneticCodeGenerator(
        population_size=100,  # Larger population for more diversity
        test_cases=test_cases
    )
    
    # Run evolution with detailed logging
    print("Starting evolution process...")
    print("Target: Create a function that sums positive numbers in a list\n")
    
    generator.evolve(
        generations=200,  # More generations for complex problems
        target_fitness=1.0  # Require perfect score
    )
    
    # Query the vector database for the best solutions
    best_solutions = generator.collection.query(
        query_texts=["sum positive numbers"],
        n_results=3,
        where={"fitness": {"$gt": 0.8}}  # Only show solutions with high fitness
    )
    
    print("\nTop 3 solutions from vector database:")
    # Check if best_solutions has expected keys
    if 'documents' in best_solutions and 'metadatas' in best_solutions:
        for i, (doc, metadata) in enumerate(zip(best_solutions['documents'], best_solutions['metadatas'])):
            # Print metadata for debugging
            print(f"Metadata for solution {i + 1}: {metadata}")
            
            if isinstance(metadata, dict):
                # Check for expected keys
                fitness = metadata.get('fitness')
                test_results = metadata.get('test_results')
                
                if fitness is not None and test_results is not None:
                    print(f"\nSolution {i + 1} (Fitness: {fitness:.2f}):")
                    print(doc)  # Assuming doc is the code as a string
                    print("\nTest Results:")
                    for result in test_results:
                        print(result)
                else:
                    print(f"Metadata for solution {i + 1} is missing 'fitness' or 'test_results'.")
            else:
                print(f"Metadata for solution {i + 1} is not a dictionary.")
    else:
        print("No valid solutions found in the vector database.")

# Additional example: Generate a function that processes strings
def string_processing_example():
    # Test cases for string processing
    string_test_cases = [
        TestCase(
            input_data={"text": "hello world"},
            expected_output="HELLO WORLD",
            test_name="Basic uppercase conversion"
        ),
        TestCase(
            input_data={"text": ""},
            expected_output="",
            test_name="Empty string"
        ),
        TestCase(
            input_data={"text": "12345"},
            expected_output="12345",
            test_name="Numbers only"
        ),
        TestCase(
            input_data={"text": "Hello World!"},
            expected_output="HELLO WORLD!",
            test_name="Mixed case with punctuation"
        )
    ]
    
    generator = GeneticCodeGenerator(
        population_size=75,
        test_cases=string_test_cases
    )
    
    print("Starting string processing evolution...")
    generator.evolve(
        generations=150,
        target_fitness=0.95
    )

# Example of generating a mathematical function
def math_function_example():
    # Test cases for a function that squares numbers and sums them
    math_test_cases = [
        TestCase(
            input_data={"numbers": [1, 2, 3]},
            expected_output=14,  # 1^2 + 2^2 + 3^2
            test_name="Simple squares"
        ),
        TestCase(
            input_data={"numbers": [-1, 2, -3]},
            expected_output=14,  # 1^2 + 2^2 + 3^2
            test_name="Handle negative numbers"
        ),
        TestCase(
            input_data={"numbers": [0]},
            expected_output=0,
            test_name="Zero input"
        ),
        TestCase(
            input_data={"numbers": []},
            expected_output=0,
            test_name="Empty list"
        ),
        TestCase(
            input_data={"numbers": [10, 10]},
            expected_output=200,  # 10^2 + 10^2
            test_name="Larger numbers"
        )
    ]
    
    generator = GeneticCodeGenerator(
        population_size=100,
        test_cases=math_test_cases
    )
    
    print("Starting mathematical function evolution...")
    generator.evolve(
        generations=150,
        target_fitness=1.0
    )

if __name__ == "__main__":
    # Run different examples
    print("=== Example 1: Sum of Positive Numbers ===")
    main()
    
    print("\n=== Example 2: String Processing ===")
    string_processing_example()
    
    print("\n=== Example 3: Mathematical Function ===")
    math_function_example()
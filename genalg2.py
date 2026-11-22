import random
import string
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
import ast
import inspect
from collections import defaultdict
import coverage
import math
from functools import wraps
import timeout_decorator

@dataclass
class TestGenome:
    """Represents a single test case genome"""
    input_data: Dict[str, Any]
    expected_output: Any
    description: str
    
    def mutate(self, mutation_rate: float = 0.1) -> None:
        """Mutate the test case inputs"""
        if random.random() < mutation_rate:
            self._mutate_input_data()
    
    def _mutate_input_data(self):
        """Mutate specific input data based on its type"""
        for key, value in self.input_data.items():
            if isinstance(value, list):
                self._mutate_list(key)
            elif isinstance(value, int):
                self._mutate_integer(key)
            elif isinstance(value, str):
                self._mutate_string(key)
            elif isinstance(value, dict):
                self._mutate_dict(key)
    
    def _mutate_list(self, key: str):
        """Mutate a list input"""
        operations = [
            'add', 'remove', 'modify', 'shuffle',
            'clear', 'duplicate', 'reverse'
        ]
        operation = random.choice(operations)
        
        if not self.input_data[key]:
            operation = 'add'
        
        if operation == 'add':
            self.input_data[key].append(self._generate_random_value())
        elif operation == 'remove':
            self.input_data[key].pop(random.randint(0, len(self.input_data[key]) - 1))
        elif operation == 'modify':
            if self.input_data[key]:
                idx = random.randint(0, len(self.input_data[key]) - 1)
                self.input_data[key][idx] = self._generate_random_value()
        elif operation == 'shuffle':
            random.shuffle(self.input_data[key])
        elif operation == 'clear':
            self.input_data[key].clear()
        elif operation == 'duplicate':
            if self.input_data[key]:
                idx = random.randint(0, len(self.input_data[key]) - 1)
                self.input_data[key].append(self.input_data[key][idx])
        elif operation == 'reverse':
            self.input_data[key].reverse()
    
    def _mutate_integer(self, key: str):
        """Mutate an integer input"""
        operations = ['increment', 'decrement', 'multiply', 'divide', 'negate', 'special']
        operation = random.choice(operations)
        
        if operation == 'increment':
            self.input_data[key] += random.randint(1, 10)
        elif operation == 'decrement':
            self.input_data[key] -= random.randint(1, 10)
        elif operation == 'multiply':
            self.input_data[key] *= random.randint(2, 5)
        elif operation == 'divide':
            if self.input_data[key] != 0:
                self.input_data[key] //= random.randint(2, 5)
        elif operation == 'negate':
            self.input_data[key] = -self.input_data[key]
        elif operation == 'special':
            self.input_data[key] = random.choice([0, 1, -1, math.inf, -math.inf])
    
    def _mutate_string(self, key: str):
        """Mutate a string input"""
        operations = [
            'append', 'prepend', 'replace', 'remove',
            'case', 'special', 'clear'
        ]
        operation = random.choice(operations)
        
        if not self.input_data[key] and operation != 'append':
            operation = 'append'
        
        if operation == 'append':
            self.input_data[key] += random.choice(string.ascii_letters + string.digits)
        elif operation == 'prepend':
            self.input_data[key] = random.choice(string.ascii_letters + string.digits) + self.input_data[key]
        elif operation == 'replace':
            if self.input_data[key]:
                pos = random.randint(0, len(self.input_data[key]) - 1)
                chars = list(self.input_data[key])
                chars[pos] = random.choice(string.ascii_letters + string.digits)
                self.input_data[key] = ''.join(chars)
        elif operation == 'remove':
            if self.input_data[key]:
                pos = random.randint(0, len(self.input_data[key]) - 1)
                self.input_data[key] = self.input_data[key][:pos] + self.input_data[key][pos + 1:]
        elif operation == 'case':
            if random.random() < 0.5:
                self.input_data[key] = self.input_data[key].upper()
            else:
                self.input_data[key] = self.input_data[key].lower()
        elif operation == 'special':
            self.input_data[key] = random.choice(['', ' ', '\n', '\t', '!@#$%'])
        elif operation == 'clear':
            self.input_data[key] = ''
    
    def _mutate_dict(self, key: str):
        """Mutate a dictionary input"""
        operations = ['add', 'remove', 'modify', 'clear']
        operation = random.choice(operations)
        
        if not self.input_data[key] and operation != 'add':
            operation = 'add'
            
        if operation == 'add':
            new_key = random.choice(string.ascii_lowercase)
            self.input_data[key][new_key] = self._generate_random_value()
        elif operation == 'remove':
            if self.input_data[key]:
                del_key = random.choice(list(self.input_data[key].keys()))
                del self.input_data[key][del_key]
        elif operation == 'modify':
            if self.input_data[key]:
                mod_key = random.choice(list(self.input_data[key].keys()))
                self.input_data[key][mod_key] = self._generate_random_value()
        elif operation == 'clear':
            self.input_data[key].clear()
    
    @staticmethod
    def _generate_random_value():
        """Generate a random value of various types"""
        value_type = random.choice(['int', 'str', 'list', 'special'])
        
        if value_type == 'int':
            return random.randint(-100, 100)
        elif value_type == 'str':
            length = random.randint(1, 10)
            return ''.join(random.choice(string.ascii_letters) for _ in range(length))
        elif value_type == 'list':
            length = random.randint(0, 5)
            return [random.randint(-10, 10) for _ in range(length)]
        else:  # special
            return random.choice([0, 1, -1, '', [], None, True, False])

class CodeCoverageAnalyzer:
    """Analyzes code coverage for test cases"""
    
    def __init__(self):
        self.cov = coverage.Coverage()
    
    def measure_coverage(self, func: callable, test_case: TestGenome) -> float:
        """Measure code coverage for a single test case"""
        self.cov.start()
        try:
            func(**test_case.input_data)
        except:
            pass
        self.cov.stop()
        
        # Get coverage data
        coverage_data = self.cov.get_data()
        total_lines = 0
        covered_lines = 0
        
        for filename in coverage_data.measured_files():
            try:
                total_lines += len(coverage_data.lines(filename))
                covered_lines += len(coverage_data.lines(filename))
            except:
                pass
        
        self.cov.erase()
        return covered_lines / total_lines if total_lines > 0 else 0.0

class TestGenerator:
    """Generates and evolves test cases for a given function"""
    
    def __init__(self, 
                 target_function: callable,
                 population_size: int = 50,
                 mutation_rate: float = 0.1):
        self.target_function = target_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.coverage_analyzer = CodeCoverageAnalyzer()
        self.best_test_cases: List[TestGenome] = []
        
        # Extract function signature
        self.signature = inspect.signature(target_function)
        self.param_types = self._infer_parameter_types()
    
    def _infer_parameter_types(self) -> Dict[str, type]:
        """Infer parameter types from function annotations and code analysis"""
        param_types = {}
        
        # Check function annotations
        for param_name, param in self.signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = param.annotation
            else:
                # Default to inferring type from function body
                param_types[param_name] = self._analyze_parameter_usage(param_name)
        
        return param_types
    
    def _analyze_parameter_usage(self, param_name: str) -> type:
        """Analyze how a parameter is used in the function body"""
        source = inspect.getsource(self.target_function)
        tree = ast.parse(source)
        
        type_hints = {
            ast.List: list,
            ast.Dict: dict,
            ast.Str: str,
            ast.Num: int,
            ast.NameConstant: bool,
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == param_name:
                parent = next((p for p in ast.walk(tree) if node in ast.iter_child_nodes(p)), None)
                if parent and type(parent) in type_hints:
                    return type_hints[type(parent)]
        
        return Any  # Default to Any if type cannot be inferred
    
    def _evaluate_test_case(self, test_case):
        """
        Evaluates the fitness of a test case based on coverage and other metrics.
        Returns a fitness score.
        """
        try:
            result = self.execute_test_case(test_case)
            # Add your fitness calculation logic here
            # Example: coverage, boundary values, edge cases
            fitness = 0
            
            # Example criteria:
            # 1. Coverage of boundary values
            if any(self._is_boundary_value(value) for value in test_case):
                fitness += 1
                
            # 2. Variation in inputs
            if self._has_input_variation(test_case):
                fitness += 1
                
            # 3. Result differentiation
            if result is not None:
                fitness += 1
                
            return fitness
            
        except Exception as e:
            # If test case causes an error, it might be interesting for testing
            # but we don't want to prioritize error cases too highly
            return 0.5
        
    def _calculate_similarity(self, test1: TestGenome, test2: TestGenome) -> float:
        """Calculate similarity between two test cases"""
        similarity = 0.0
        total_comparisons = 0
        
        for key in test1.input_data:
            if key in test2.input_data:
                value1 = test1.input_data[key]
                value2 = test2.input_data[key]
                
                if isinstance(value1, list) and isinstance(value2, list):
                    # Compare list lengths and contents
                    len_sim = 1.0 - abs(len(value1) - len(value2)) / max(len(value1) + len(value2), 1)
                    content_sim = len(set(value1) & set(value2)) / max(len(set(value1) | set(value2)), 1)
                    similarity += (len_sim + content_sim) / 2
                
                elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Compare numerical values
                    max_diff = max(abs(value1), abs(value2), 1)
                    similarity += 1.0 - min(1.0, abs(value1 - value2) / max_diff)
                
                elif isinstance(value1, str) and isinstance(value2, str):
                    # Compare string lengths and contents
                    len_sim = 1.0 - abs(len(value1) - len(value2)) / max(len(value1) + len(value2), 1)
                    content_sim = sum(a == b for a, b in zip(value1, value2)) / max(len(value1), len(value2), 1)
                    similarity += (len_sim + content_sim) / 2
                
                elif isinstance(value1, dict) and isinstance(value2, dict):
                    # Compare dictionary keys and values
                    keys_sim = len(set(value1.keys()) & set(value2.keys())) / max(len(set(value1.keys()) | set(value2.keys())), 1)
                    similarity += keys_sim
                
                total_comparisons += 1
        
        return similarity / total_comparisons if total_comparisons > 0 else 0.0

    def generate_test_suite(self, generations: int = 50) -> List[TestGenome]:
        """Generate a suite of test cases using genetic algorithm"""
        # Initialize population
        population = [self._generate_random_test_case() for _ in range(self.population_size)]
        
        for generation in range(generations):
            # Evaluate fitness for all test cases
            fitness_scores = [(test_case, self._evaluate_test_case(test_case))
                            for test_case in population]
            
            # Sort by fitness and select top performers
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            elite = [test_case for test_case, _ in fitness_scores[:self.population_size // 4]]
            
            # Create new population
            new_population = elite.copy()
            
            # Add mutated versions of elite
            for test_case in elite:
                mutated = TestGenome(
                    input_data=test_case.input_data.copy(),
                    expected_output=test_case.expected_output,
                    description=test_case.description
                )
                mutated.mutate(self.mutation_rate)
                new_population.append(mutated)
            
            # Fill remaining population with new random test cases
            while len(new_population) < self.population_size:
                new_population.append(self._generate_random_test_case())
            
            population = new_population
            
            # Update best test cases
            self._update_best_test_cases(fitness_scores)
        
        return self.best_test_cases
    def _generate_random_test_case(self) -> TestGenome:
        """Generate a random test case based on function signature"""
        input_data = {}
        
        for param_name, param in self.signature.parameters.items():
            param_type = self.param_types.get(param_name, Any)
            
            # Generate value based on type
            if param_type == int:
                input_data[param_name] = random.randint(-100, 100)
            elif param_type == float:
                input_data[param_name] = random.uniform(-100.0, 100.0)
            elif param_type == str:
                length = random.randint(0, 20)
                input_data[param_name] = ''.join(random.choice(string.ascii_letters) for _ in range(length))
            elif param_type == bool:
                input_data[param_name] = random.choice([True, False])
            elif param_type == list:
                length = random.randint(0, 10)
                input_data[param_name] = [random.randint(-100, 100) for _ in range(length)]
            elif param_type == dict:
                size = random.randint(0, 5)
                input_data[param_name] = {
                    random.choice(string.ascii_lowercase): random.randint(-10, 10)
                    for _ in range(size)
                }
            else:
                # For Any type or unknown types, try different value types
                value_type = random.choice(['int', 'float', 'str', 'bool', 'list', 'dict', 'none'])
                if value_type == 'int':
                    input_data[param_name] = random.randint(-100, 100)
                elif value_type == 'float':
                    input_data[param_name] = random.uniform(-100.0, 100.0)
                elif value_type == 'str':
                    input_data[param_name] = ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(0, 20)))
                elif value_type == 'bool':
                    input_data[param_name] = random.choice([True, False])
                elif value_type == 'list':
                    input_data[param_name] = [random.randint(-10, 10) for _ in range(random.randint(0, 5))]
                elif value_type == 'dict':
                    input_data[param_name] = {str(i): random.randint(-10, 10) for i in range(random.randint(0, 3))}
                else:  # none
                    input_data[param_name] = None

            # Handle default values
            if param.default != inspect.Parameter.empty and random.random() < 0.3:
                input_data[param_name] = param.default

        # Try to execute the function to get expected output
        try:
            expected_output = self.target_function(**input_data)
        except Exception as e:
            expected_output = e  # Store exception as expected output for error cases

        return TestGenome(
            input_data=input_data,
            expected_output=expected_output,
            description=self._generate_test_description(input_data)
        )

    def _generate_test_description(self, input_data: Dict[str, Any]) -> str:
        """Generate a human-readable description of the test case"""
        descriptions = []
        
        for param_name, value in input_data.items():
            if isinstance(value, list):
                descriptions.append(f"{param_name}: list of length {len(value)}")
            elif isinstance(value, str):
                descriptions.append(f"{param_name}: string of length {len(value)}")
            elif isinstance(value, dict):
                descriptions.append(f"{param_name}: dict with {len(value)} items")
            else:
                descriptions.append(f"{param_name}: {value}")
        
        return ", ".join(descriptions)
    
    def _update_best_test_cases(self, fitness_scores: List[Tuple[TestGenome, float]]) -> None:
            """Update the collection of best test cases"""
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add new high-performing test cases
            for test_case, fitness in fitness_scores[:5]:  # Consider top 5 from current generation
                if not self.best_test_cases or fitness > 0.8:  # High fitness threshold
                    if not any(self._calculate_similarity(test_case, existing) > 0.9 
                              for existing in self.best_test_cases):
                        self.best_test_cases.append(test_case)
            
            # Limit size of best test cases collection
            max_best_cases = 20
            if len(self.best_test_cases) > max_best_cases:
                # Sort by fitness and keep top cases
                best_scores = [(test_case, self._evaluate_test_case(test_case))
                              for test_case in self.best_test_cases]
                best_scores.sort(key=lambda x: x[1], reverse=True)
                self.best_test_cases = [test_case for test_case, _ in best_scores[:max_best_cases]]

    @timeout_decorator.timeout(5)  # 5 second timeout for test execution
    def execute_test_case(self, test_case: TestGenome) -> Tuple[bool, Any, Optional[Exception]]:
            """Execute a single test case and return results"""
            try:
                actual_output = self.target_function(**test_case.input_data)
                is_success = actual_output == test_case.expected_output
                return is_success, actual_output, None
            except Exception as e:
                return False, None, e

    def execute_test_suite(self) -> Dict[str, Any]:
            """Execute all test cases in the test suite and return results"""
            results = {
                'total_tests': len(self.best_test_cases),
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'details': []
            }

            for test_case in self.best_test_cases:
                try:
                    is_success, actual_output, error = self.execute_test_case(test_case)
                    
                    test_result = {
                        'description': test_case.description,
                        'inputs': test_case.input_data,
                        'expected': test_case.expected_output,
                        'actual': actual_output,
                        'status': 'PASS' if is_success else 'FAIL',
                        'error': str(error) if error else None
                    }

                    if error:
                        results['errors'] += 1
                    elif is_success:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1

                    results['details'].append(test_result)

                except timeout_decorator.TimeoutError:
                    results['errors'] += 1
                    results['details'].append({
                        'description': test_case.description,
                        'inputs': test_case.input_data,
                        'expected': test_case.expected_output,
                        'actual': None,
                        'status': 'ERROR',
                        'error': 'Test execution timeout'
                    })

            results['coverage'] = self._calculate_suite_coverage()
            return results

    def _calculate_suite_coverage(self) -> float:
            """Calculate the total code coverage for the entire test suite"""
            self.coverage_analyzer.cov.start()
            
            for test_case in self.best_test_cases:
                try:
                    self.target_function(**test_case.input_data)
                except:
                    continue
                    
            self.coverage_analyzer.cov.stop()
            
            # Get coverage data
            coverage_data = self.coverage_analyzer.cov.get_data()
            total_lines = 0
            covered_lines = 0
            
            for filename in coverage_data.measured_files():
                try:
                    total_lines += len(coverage_data.lines(filename))
                    covered_lines += len(coverage_data.lines(filename))
                except:
                    continue
            
            self.coverage_analyzer.cov.erase()
            return covered_lines / total_lines if total_lines > 0 else 0.0

    def get_test_suite_summary(self) -> str:
            """Generate a human-readable summary of the test suite"""
            results = self.execute_test_suite()
            
            summary = [
                "Test Suite Summary",
                "=" * 20,
                f"Total Tests: {results['total_tests']}",
                f"Passed: {results['passed']}",
                f"Failed: {results['failed']}",
                f"Errors: {results['errors']}",
                f"Coverage: {results['coverage']*100:.1f}%",
                "\nDetailed Results:",
                "=" * 20
            ]

            for i, test in enumerate(results['details'], 1):
                summary.extend([
                    f"\nTest #{i}: {test['description']}",
                    f"Status: {test['status']}",
                    f"Inputs: {test['inputs']}",
                    f"Expected: {test['expected']}",
                    f"Actual: {test['actual']}"
                ])
                
                if test['error']:
                    summary.append(f"Error: {test['error']}")

            return "\n".join(summary)

def generate_tests(func: callable, 
                  population_size: int = 50,
                  generations: int = 50,
                  mutation_rate: float = 0.1) -> TestGenerator:
    """
    Convenience function to generate tests for a given function.
    
    Args:
        func: The function to generate tests for
        population_size: Size of the test case population
        generations: Number of generations to evolve
        mutation_rate: Rate of mutation for test cases
        
    Returns:
        TestGenerator instance with generated test cases
    """
    generator = TestGenerator(
        target_function=func,
        population_size=population_size,
        mutation_rate=mutation_rate
    )
    generator.generate_test_suite(generations=generations)
    return generator

# Example usage:
if __name__ == "__main__":
    # Example function to test
    def calculate_discount(price: float, quantity: int, is_vip: bool = False) -> float:
        """Calculate discount for a purchase"""
        if price <= 0 or quantity <= 0:
            raise ValueError("Price and quantity must be positive")
            
        total = price * quantity
        
        if is_vip:
            if total > 1000:
                return total * 0.2  # 20% discount
            else:
                return total * 0.1  # 10% discount
        else:
            if total > 500:
                return total * 0.05  # 5% discount
            else:
                return 0  # No discount
    
    # Generate tests
    generator = generate_tests(calculate_discount)
    
    # Print test suite summary
    print(generator.get_test_suite_summary())
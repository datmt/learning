# Building Self-Improving Agents with Reflexion
## Learn from Failures, Improve Over Time

> **Goal**: Build an agent that reflects on its mistakes, stores learnings in memory, and improves performance through self-critique and iterative refinement.

---

## Part 1: What is Reflexion?

### The Problem with Standard Agents

Traditional agents make mistakes and keep making the same mistakes:

```
Task: "Book a flight from NYC to London"

Attempt 1: Calls API with wrong date format → Fails
Attempt 2: Calls API with wrong date format again → Fails
Attempt 3: Calls API with wrong date format again → Fails

Problem: No learning between attempts!
```

### Reflexion: Learning from Mistakes

**Reflexion** adds a self-reflection loop:

```
Task: "Book a flight from NYC to London"

Attempt 1: 
  → Action: Call API with MM/DD/YYYY
  → Result: Error "Invalid format"
  → Reflection: "API requires YYYY-MM-DD format, not MM/DD/YYYY"
  → Store in memory: "Flight API date format is YYYY-MM-DD"

Attempt 2:
  → Check memory: "Use YYYY-MM-DD for dates"
  → Action: Call API with YYYY-MM-DD
  → Result: Success! ✓
```

**Key insight**: Agent builds up knowledge through trial, error, and reflection.

---

## Part 2: Architecture Overview

### The Reflexion Loop

```
┌─────────────────────────────────────────────┐
│                                             │
│  1. PLAN                                    │
│     ↓                                       │
│  2. ACT (use tools)                         │
│     ↓                                       │
│  3. OBSERVE (results)                       │
│     ↓                                       │
│  4. EVALUATE (success/failure?)             │
│     ↓                                       │
│     ├── Success → Done                      │
│     └── Failure → REFLECT                   │
│         ↓                                   │
│  5. REFLECT (what went wrong?)              │
│     ↓                                       │
│  6. STORE LEARNING (in memory)              │
│     ↓                                       │
│     Back to step 1 (with new knowledge) ────┘
│                                             │
└─────────────────────────────────────────────┘
```

### Components We'll Build

1. **Planner**: Multi-step task decomposition
2. **Actor**: Execute actions with tools
3. **Evaluator**: Judge success/failure
4. **Reflector**: Analyze what went wrong
5. **Memory**: Store short-term trajectory + long-term learnings
6. **Prompt Engineer**: Reliable tool use patterns

---

## Part 3: Memory Systems

### Three Types of Memory

```python
class AgentMemory:
    """
    Three-tier memory system for learning agents.
    """
    
    def __init__(self):
        # Working memory: Current task context
        self.working_memory = {
            'task': None,
            'plan': [],
            'current_step': 0,
            'trajectory': []  # History of this attempt
        }
        
        # Episodic memory: Past attempts at similar tasks
        self.episodic_memory = []  # List of past episodes
        
        # Semantic memory: General learnings/rules
        self.semantic_memory = {
            'successful_patterns': [],
            'known_failures': [],
            'tool_guidelines': {}
        }
    
    def add_to_trajectory(self, step_type, content):
        """Add to current attempt's trajectory."""
        self.working_memory['trajectory'].append({
            'step': self.working_memory['current_step'],
            'type': step_type,  # 'thought', 'action', 'observation'
            'content': content,
            'timestamp': time.time()
        })
        self.working_memory['current_step'] += 1
    
    def reflect_on_trajectory(self):
        """Analyze current trajectory for learnings."""
        trajectory = self.working_memory['trajectory']
        
        # Extract what went wrong
        failures = [
            step for step in trajectory 
            if 'error' in step.get('content', '').lower()
        ]
        
        return failures
    
    def store_episodic(self, task, trajectory, success, reflection):
        """Store completed episode."""
        episode = {
            'task': task,
            'trajectory': trajectory,
            'success': success,
            'reflection': reflection,
            'timestamp': time.time()
        }
        self.episodic_memory.append(episode)
    
    def store_semantic(self, learning_type, content):
        """Store general learning."""
        if learning_type == 'pattern':
            self.semantic_memory['successful_patterns'].append(content)
        elif learning_type == 'failure':
            self.semantic_memory['known_failures'].append(content)
        elif learning_type == 'tool_guideline':
            tool_name = content['tool']
            if tool_name not in self.semantic_memory['tool_guidelines']:
                self.semantic_memory['tool_guidelines'][tool_name] = []
            self.semantic_memory['tool_guidelines'][tool_name].append(
                content['guideline']
            )
    
    def retrieve_relevant_memories(self, task):
        """Retrieve past learnings relevant to current task."""
        # Find similar past episodes
        relevant_episodes = [
            ep for ep in self.episodic_memory
            if self._is_similar(task, ep['task'])
        ][-3:]  # Last 3 similar episodes
        
        # Get relevant semantic memories
        relevant_patterns = self.semantic_memory['successful_patterns']
        relevant_failures = self.semantic_memory['known_failures']
        
        return {
            'episodes': relevant_episodes,
            'patterns': relevant_patterns,
            'failures': relevant_failures
        }
    
    def _is_similar(self, task1, task2):
        """Simple similarity check (can be improved with embeddings)."""
        # Basic keyword matching for now
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap > 0.3
    
    def clear_working_memory(self):
        """Reset for new task."""
        self.working_memory = {
            'task': None,
            'plan': [],
            'current_step': 0,
            'trajectory': []
        }
```

---

## Part 4: Prompt Engineering for Reliable Tool Use

### Tool Schema Definition

```python
TOOLS = {
    'search': {
        'description': 'Search the web for information',
        'parameters': {
            'query': 'str - search query',
        },
        'returns': 'str - search results',
        'examples': [
            {'query': 'weather in London', 'result': 'Temperature: 15°C...'},
            {'query': 'python list comprehension', 'result': 'List comprehensions...'}
        ],
        'common_errors': [
            'Empty query returns no results',
            'Very specific queries may return nothing - try broader terms'
        ]
    },
    'calculate': {
        'description': 'Perform mathematical calculations',
        'parameters': {
            'expression': 'str - mathematical expression (e.g., "2 + 2", "sqrt(16)")'
        },
        'returns': 'float - calculation result',
        'examples': [
            {'expression': '15 * 20', 'result': 300},
            {'expression': 'sqrt(144)', 'result': 12.0}
        ],
        'common_errors': [
            'Invalid syntax returns error - ensure valid Python math expression',
            'Division by zero returns error'
        ]
    },
    'file_read': {
        'description': 'Read contents of a file',
        'parameters': {
            'filepath': 'str - path to file'
        },
        'returns': 'str - file contents',
        'examples': [
            {'filepath': '/data/report.txt', 'result': 'Report contents...'}
        ],
        'common_errors': [
            'File not found - check path exists',
            'Permission denied - check file permissions'
        ]
    }
}
```

### Prompt Template for Tool Use

```python
def create_tool_prompt(task, available_tools, memory_context):
    """
    Create a prompt that encourages reliable tool use.
    """
    
    # Build tool descriptions with examples and error patterns
    tool_descriptions = []
    for tool_name, tool_info in available_tools.items():
        desc = f"""
Tool: {tool_name}
Description: {tool_info['description']}
Parameters: {tool_info['parameters']}
Returns: {tool_info['returns']}

Examples:
{chr(10).join(f"  - Input: {ex} → Output: {tool_info['examples']}" for ex in tool_info['examples'][:2])}

Common errors to avoid:
{chr(10).join(f"  - {err}" for err in tool_info['common_errors'])}
"""
        tool_descriptions.append(desc)
    
    # Include past learnings from memory
    learnings = ""
    if memory_context['failures']:
        learnings += "\nPast failures to avoid:\n"
        for failure in memory_context['failures'][-3:]:
            learnings += f"  - {failure}\n"
    
    if memory_context['patterns']:
        learnings += "\nSuccessful patterns:\n"
        for pattern in memory_context['patterns'][-3:]:
            learnings += f"  - {pattern}\n"
    
    prompt = f"""You are a helpful agent that can use tools to accomplish tasks.

Task: {task}

Available tools:
{chr(10).join(tool_descriptions)}

{learnings}

Instructions:
1. Think step-by-step about what you need to do
2. Choose the appropriate tool and parameters carefully
3. If a tool fails, analyze WHY it failed before retrying
4. Learn from errors and adjust your approach

Format your response as:
Thought: [your reasoning]
Action: tool_name
Action Input: {{"parameter": "value"}}

OR if task is complete:
Thought: [summary of what was accomplished]
Final Answer: [result]
"""
    
    return prompt
```

---

## Part 5: Multi-Step Planning and Decomposition

### Task Decomposition Strategy

```python
def decompose_task(task):
    """
    Break down complex task into subtasks.
    Uses LLM to plan steps.
    """
    
    planning_prompt = f"""Break down this task into clear, sequential steps.

Task: {task}

Provide a numbered plan where each step is:
1. Specific and actionable
2. Uses available tools when needed
3. Builds on previous steps
4. Has a clear success criterion

Format:
Step 1: [description] - Success: [how to verify]
Step 2: [description] - Success: [how to verify]
...

Plan:"""

    # Call LLM for planning
    response = call_llm(planning_prompt)
    
    # Parse plan into structured steps
    steps = parse_plan(response)
    
    return steps


def parse_plan(plan_text):
    """Parse LLM's plan into structured format."""
    steps = []
    
    for line in plan_text.split('\n'):
        if line.strip().startswith('Step'):
            # Extract step number, description, and success criterion
            parts = line.split('-')
            description = parts[0].split(':', 1)[1].strip()
            success = parts[1].replace('Success:', '').strip() if len(parts) > 1 else None
            
            steps.append({
                'description': description,
                'success_criterion': success,
                'status': 'pending'
            })
    
    return steps


# Example usage
task = "Find the cheapest flight from NYC to London next week and book it"

plan = decompose_task(task)
# Returns:
# [
#   {
#     'description': 'Search for flights from NYC to London for next week',
#     'success_criterion': 'Retrieved list of flights with prices',
#     'status': 'pending'
#   },
#   {
#     'description': 'Compare prices and find cheapest option',
#     'success_criterion': 'Identified cheapest flight',
#     'status': 'pending'
#   },
#   {
#     'description': 'Book the selected flight',
#     'success_criterion': 'Received booking confirmation',
#     'status': 'pending'
#   }
# ]
```

---

## Part 6: Error Handling and Recovery

### Error Classification

```python
class ErrorHandler:
    """
    Classify and handle different types of errors.
    """
    
    ERROR_TYPES = {
        'tool_not_found': {
            'pattern': r'tool.*not (found|available)',
            'recovery': 'check_available_tools',
            'severity': 'high'
        },
        'invalid_parameters': {
            'pattern': r'invalid (parameter|argument)',
            'recovery': 'fix_parameters',
            'severity': 'medium'
        },
        'api_error': {
            'pattern': r'(api|http) error|status code',
            'recovery': 'retry_with_backoff',
            'severity': 'medium'
        },
        'timeout': {
            'pattern': r'timeout|timed out',
            'recovery': 'retry_with_longer_timeout',
            'severity': 'medium'
        },
        'rate_limit': {
            'pattern': r'rate limit|too many requests',
            'recovery': 'wait_and_retry',
            'severity': 'low'
        }
    }
    
    def classify_error(self, error_message):
        """Determine error type from message."""
        import re
        
        error_message_lower = error_message.lower()
        
        for error_type, info in self.ERROR_TYPES.items():
            if re.search(info['pattern'], error_message_lower):
                return {
                    'type': error_type,
                    'recovery_strategy': info['recovery'],
                    'severity': info['severity']
                }
        
        return {
            'type': 'unknown',
            'recovery_strategy': 'reflect_and_retry',
            'severity': 'high'
        }
    
    def handle_error(self, error_info, context):
        """Execute recovery strategy."""
        strategy = error_info['recovery_strategy']
        
        if strategy == 'check_available_tools':
            return self._check_tools(context)
        elif strategy == 'fix_parameters':
            return self._fix_parameters(context)
        elif strategy == 'retry_with_backoff':
            return self._retry_with_backoff(context)
        elif strategy == 'wait_and_retry':
            return self._wait_and_retry(context)
        else:
            return self._reflect_and_retry(context)
    
    def _check_tools(self, context):
        """Verify tool exists and suggest alternatives."""
        available = list(TOOLS.keys())
        return {
            'action': 'inform',
            'message': f"Available tools: {', '.join(available)}. Please choose from these."
        }
    
    def _fix_parameters(self, context):
        """Analyze and fix parameter issues."""
        tool_name = context.get('tool_name')
        if tool_name in TOOLS:
            expected_params = TOOLS[tool_name]['parameters']
            return {
                'action': 'retry',
                'guidance': f"Expected parameters: {expected_params}. Check format and types."
            }
        return {'action': 'reflect'}
    
    def _retry_with_backoff(self, context):
        """Retry with exponential backoff."""
        attempt = context.get('attempt', 0)
        wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
        
        return {
            'action': 'wait_and_retry',
            'wait_time': wait_time
        }
    
    def _wait_and_retry(self, context):
        """Wait before retrying (for rate limits)."""
        return {
            'action': 'wait_and_retry',
            'wait_time': 60  # Wait 1 minute for rate limits
        }
    
    def _reflect_and_retry(self, context):
        """Trigger reflection on unknown errors."""
        return {
            'action': 'reflect',
            'message': 'Unexpected error - analyze what went wrong before retrying'
        }
```

### Retry Logic with Learning

```python
def execute_with_retry(action, max_attempts=3, memory=None):
    """
    Execute action with intelligent retry and learning.
    """
    error_handler = ErrorHandler()
    
    for attempt in range(max_attempts):
        try:
            # Execute action
            result = execute_action(action)
            
            # Success! Store pattern
            if memory:
                memory.store_semantic('pattern', {
                    'action': action,
                    'context': 'This approach worked',
                    'success': True
                })
            
            return result
            
        except Exception as e:
            error_message = str(e)
            
            # Classify error
            error_info = error_handler.classify_error(error_message)
            
            print(f"Attempt {attempt + 1} failed: {error_info['type']}")
            
            # Last attempt - reflect and store learning
            if attempt == max_attempts - 1:
                if memory:
                    reflection = reflect_on_failure(action, error_message, attempt + 1)
                    memory.store_semantic('failure', {
                        'action': action,
                        'error': error_message,
                        'reflection': reflection
                    })
                raise
            
            # Determine recovery strategy
            recovery = error_handler.handle_error(
                error_info,
                {'tool_name': action.get('tool'), 'attempt': attempt}
            )
            
            # Execute recovery
            if recovery['action'] == 'wait_and_retry':
                time.sleep(recovery['wait_time'])
            elif recovery['action'] == 'reflect':
                # Trigger reflection before retry
                reflection = reflect_on_failure(action, error_message, attempt + 1)
                print(f"Reflection: {reflection}")
                # Potentially modify action based on reflection
                action = apply_reflection_to_action(action, reflection)
    
    raise Exception("Max attempts exceeded")
```

---

## Part 7: The Reflection Module

### Generating Meaningful Reflections

```python
def reflect_on_failure(action, error, trajectory, memory):
    """
    Generate reflection on why action failed.
    This is the core of Reflexion.
    """
    
    # Retrieve relevant past failures
    similar_failures = [
        f for f in memory.semantic_memory['known_failures']
        if f.get('action', {}).get('tool') == action.get('tool')
    ]
    
    reflection_prompt = f"""Analyze this failure and provide insights for improvement.

Action attempted:
{json.dumps(action, indent=2)}

Error received:
{error}

Recent trajectory:
{format_trajectory(trajectory[-5:])}

Similar past failures:
{chr(10).join(f"- {f['reflection']}" for f in similar_failures[-3:])}

Provide a reflection that answers:
1. What specifically went wrong?
2. Why did it go wrong? (root cause)
3. How should we approach this differently next time?
4. What general principle can we extract?

Reflection:"""

    reflection = call_llm(reflection_prompt)
    
    # Extract actionable insight
    insight = extract_insight(reflection)
    
    return {
        'full_reflection': reflection,
        'actionable_insight': insight,
        'context': {
            'action': action,
            'error': error
        }
    }


def extract_insight(reflection):
    """Extract concrete, actionable insight from reflection."""
    
    extraction_prompt = f"""From this reflection, extract ONE specific, actionable guideline.

Reflection:
{reflection}

Guideline format: "When [situation], [action] because [reason]"

Guideline:"""

    insight = call_llm(extraction_prompt)
    
    return insight.strip()


# Example
action = {'tool': 'calculate', 'input': {'expression': '10 / 0'}}
error = "ZeroDivisionError: division by zero"

reflection = reflect_on_failure(action, error, trajectory, memory)
# Returns:
# {
#   'full_reflection': 'The calculation failed because we attempted to divide by zero...',
#   'actionable_insight': 'When using calculate tool, validate that divisor is not zero before execution',
#   'context': {...}
# }
```

---

## Part 8: Complete Reflexion Agent

```python
class ReflexionAgent:
    """
    Self-improving agent with reflection capabilities.
    """
    
    def __init__(self, tools, llm_caller):
        self.tools = tools
        self.llm = llm_caller
        self.memory = AgentMemory()
        self.error_handler = ErrorHandler()
    
    def run(self, task, max_iterations=10):
        """
        Execute task with reflection loop.
        """
        print(f"\n{'='*70}")
        print(f"TASK: {task}")
        print(f"{'='*70}\n")
        
        self.memory.clear_working_memory()
        self.memory.working_memory['task'] = task
        
        # Retrieve relevant memories
        memory_context = self.memory.retrieve_relevant_memories(task)
        
        # Plan task
        plan = self._plan(task, memory_context)
        self.memory.working_memory['plan'] = plan
        
        # Execute plan with reflection
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Act
            result = self._act(task, memory_context)
            
            # Evaluate
            evaluation = self._evaluate(task, result)
            
            if evaluation['success']:
                print(f"\n✓ Task completed successfully!")
                
                # Store successful episode
                self.memory.store_episodic(
                    task=task,
                    trajectory=self.memory.working_memory['trajectory'],
                    success=True,
                    reflection="Task completed successfully"
                )
                
                return result
            
            # Reflect on failure
            print(f"\n✗ Iteration failed. Reflecting...")
            reflection = self._reflect(
                task,
                result,
                evaluation
            )
            
            print(f"Reflection: {reflection['actionable_insight']}")
            
            # Store learning
            self.memory.store_semantic('failure', reflection)
            
            # Update memory context for next iteration
            memory_context = self.memory.retrieve_relevant_memories(task)
        
        print(f"\n✗ Max iterations reached. Task incomplete.")
        return None
    
    def _plan(self, task, memory_context):
        """Create execution plan."""
        prompt = f"""Create a plan to accomplish this task.

Task: {task}

Available tools: {list(self.tools.keys())}

Past learnings:
{self._format_memory_context(memory_context)}

Provide a step-by-step plan:"""

        plan_text = self.llm(prompt)
        plan = parse_plan(plan_text)
        
        print(f"Plan created:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step['description']}")
        
        return plan
    
    def _act(self, task, memory_context):
        """Execute next action."""
        prompt = create_tool_prompt(task, self.tools, memory_context)
        
        # Add current trajectory to prompt
        if self.memory.working_memory['trajectory']:
            prompt += f"\n\nPrevious actions:\n"
            prompt += self._format_trajectory(
                self.memory.working_memory['trajectory']
            )
        
        response = self.llm(prompt)
        
        # Parse response
        action = self._parse_action(response)
        
        if action is None:
            # Final answer
            return {'type': 'final_answer', 'content': response}
        
        # Execute tool
        try:
            result = self._execute_tool(action)
            
            self.memory.add_to_trajectory('action', action)
            self.memory.add_to_trajectory('observation', result)
            
            return {'type': 'tool_result', 'content': result}
            
        except Exception as e:
            error_info = self.error_handler.classify_error(str(e))
            
            self.memory.add_to_trajectory('error', {
                'action': action,
                'error': str(e),
                'type': error_info['type']
            })
            
            return {'type': 'error', 'content': str(e), 'error_info': error_info}
    
    def _evaluate(self, task, result):
        """Evaluate if task is complete and successful."""
        
        if result['type'] == 'final_answer':
            # Check if answer actually solves the task
            eval_prompt = f"""Did this answer successfully complete the task?

Task: {task}

Answer: {result['content']}

Respond with:
Success: true/false
Reason: [why it succeeded or failed]"""

            eval_response = self.llm(eval_prompt)
            
            return self._parse_evaluation(eval_response)
        
        elif result['type'] == 'error':
            return {
                'success': False,
                'reason': f"Error occurred: {result['content']}"
            }
        
        else:
            # Tool execution - not yet complete
            return {
                'success': False,
                'reason': 'Task in progress'
            }
    
    def _reflect(self, task, result, evaluation):
        """Generate reflection on failure."""
        
        trajectory = self.memory.working_memory['trajectory']
        
        reflection = reflect_on_failure(
            action=trajectory[-2] if len(trajectory) >= 2 else {},
            error=result.get('content', ''),
            trajectory=trajectory,
            memory=self.memory
        )
        
        return reflection
    
    def _execute_tool(self, action):
        """Execute tool call."""
        tool_name = action['tool']
        tool_input = action['input']
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_func = self.tools[tool_name]['function']
        result = tool_func(**tool_input)
        
        return result
    
    def _parse_action(self, response):
        """Parse LLM response for action."""
        import re
        
        # Look for Action: and Action Input: patterns
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*({.*?})', response, re.DOTALL)
        
        if action_match and input_match:
            import json
            return {
                'tool': action_match.group(1),
                'input': json.loads(input_match.group(1))
            }
        
        return None  # No action found - likely final answer
    
    def _format_memory_context(self, context):
        """Format memory context for prompt."""
        output = []
        
        if context['failures']:
            output.append("Past failures:")
            for f in context['failures'][-3:]:
                output.append(f"  - {f['reflection']}")
        
        if context['patterns']:
            output.append("Successful patterns:")
            for p in context['patterns'][-3:]:
                output.append(f"  - {p}")
        
        return '\n'.join(output) if output else "No past learnings"
    
    def _format_trajectory(self, trajectory):
        """Format trajectory for display."""
        output = []
        for step in trajectory:
            output.append(f"{step['type']}: {step['content']}")
        return '\n'.join(output)
    
    def _parse_evaluation(self, eval_text):
        """Parse evaluation response."""
        success = 'true' in eval_text.lower()
        
        reason_match = re.search(r'Reason:\s*(.+)', eval_text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "Unknown"
        
        return {
            'success': success,
            'reason': reason
        }
```

---

## Part 9: Example Tools Implementation

```python
import math
import time

def create_example_tools():
    """Create example tools for demonstration."""
    
    tools = {
        'search': {
            'description': 'Search for information',
            'function': lambda query: f"Search results for '{query}': [Simulated results...]",
            'parameters': {'query': 'str'}
        },
        'calculate': {
            'description': 'Perform calculations',
            'function': lambda expression: eval(expression),  # Note: unsafe, just for demo
            'parameters': {'expression': 'str'}
        },
        'get_time': {
            'description': 'Get current time',
            'function': lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {}
        },
        'add_numbers': {
            'description': 'Add two numbers',
            'function': lambda a, b: a + b,
            'parameters': {'a': 'int', 'b': 'int'}
        }
    }
    
    return tools
```

---

## Part 10: Putting It All Together

```python
def demo_reflexion_agent():
    """Demonstrate the reflexion agent."""
    
    # Setup
    tools = create_example_tools()
    
    def simple_llm(prompt):
        """Simple mock LLM for demonstration."""
        # In real implementation, call actual LLM API
        print(f"\n[LLM called with prompt length: {len(prompt)}]")
        
        # Simulate response
        if "calculate" in prompt.lower() and "15 * 20" in prompt:
            return """Thought: I need to multiply 15 by 20.
Action: calculate
Action Input: {"expression": "15 * 20"}"""
        
        return "Final Answer: Task complete"
    
    # Create agent
    agent = ReflexionAgent(tools, simple_llm)
    
    # Run task
    task = "Calculate 15 times 20"
    result = agent.run(task, max_iterations=5)
    
    print(f"\nFinal result: {result}")
    
    # Show what was learned
    print(f"\n{'='*70}")
    print("MEMORY SUMMARY")
    print(f"{'='*70}")
    print(f"Episodic memories: {len(agent.memory.episodic_memory)}")
    print(f"Semantic learnings: {len(agent.memory.semantic_memory['known_failures'])}")


if __name__ == "__main__":
    demo_reflexion_agent()
```

---

## Part 11: Key Takeaways

### What You've Built

1. **Three-tier memory system**
   - Working: Current task
   - Episodic: Past attempts
   - Semantic: General learnings

2. **Structured error handling**
   - Classification by type
   - Recovery strategies
   - Learning from failures

3. **Reliable tool use**
   - Schema-driven prompts
   - Error examples in context
   - Past learnings guide future actions

4. **Multi-step planning**
   - Task decomposition
   - Success criteria
   - Progressive execution

5. **Self-improvement loop**
   - Reflect on failures
   - Extract insights
   - Store for future use

### Why This Matters

**Traditional agent**: Makes same mistake repeatedly
**Reflexion agent**: Each failure makes it smarter

**Real-world impact**:
- Fewer API calls (learns from mistakes)
- Better success rate (accumulates knowledge)
- Generalizes learnings (applies patterns to new tasks)

---

## Part 12: Extensions and Improvements

### Advanced Enhancements

1. **Vector-based memory retrieval**
```python
from sentence_transformers import SentenceTransformer

class VectorMemory(AgentMemory):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.episode_embeddings = []
    
    def retrieve_relevant_memories(self, task):
        task_embedding = self.encoder.encode(task)
        
        # Compute similarity with all episodes
        similarities = [
            cosine_similarity(task_embedding, ep_emb)
            for ep_emb in self.episode_embeddings
        ]
        
        # Return top-k most similar
        top_k_idx = np.argsort(similarities)[-3:]
        relevant_episodes = [self.episodic_memory[i] for i in top_k_idx]
        
        return {'episodes': relevant_episodes, ...}
```

2. **Reflection quality scoring**
```python
def score_reflection_quality(reflection, outcome):
    """
    Score how useful a reflection was.
    Update future retrieval based on usefulness.
    """
    # Track if following this reflection led to success
    return quality_score
```

3. **Multi-agent reflection**
```python
def collaborative_reflection(agent1_trajectory, agent2_trajectory):
    """
    Multiple agents reflect on each other's approaches.
    """
    # Compare strategies
    # Extract best practices
    # Share learnings
```

---

## Resources

- **Reflexion Paper**: "Reflexion: Language Agents with Verbal Reinforcement Learning"
- **ReAct Paper**: "ReAct: Synergizing Reasoning and Acting in Language Models"
- **LangChain Agents**: https://python.langchain.com/docs/modules/agents/
- **AutoGPT**: Example of self-improving agent architecture

---

## Your Assignment

1. **Implement the agent** with 3-5 simple tools
2. **Test on a task that requires multiple attempts** (e.g., API with specific format requirements)
3. **Watch it learn** - run the same task twice, see improvement
4. **Extend with one enhancement** (vector memory, reflection scoring, or your idea)
5. **Answer**: What types of tasks benefit most from Reflexion? Why?

This is where everything comes together - your understanding of prompting, error handling, and now self-improvement creates truly capable agents!
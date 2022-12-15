from violations import Violation
from typing import List

class ViolationCache:
    def __init__(self, violations:List[Violation]):
        self.violations = violations
        self.cache = {}

    def _hash(self, observation, proposed_action):
        # observation and proposed action are numpy arrays, make them hashable
        return hash((observation.tostring(), proposed_action.tostring()))

    def in_violation(self, observation, proposed_action) -> bool:
        if self._hash(observation, proposed_action) in self.cache:
            return self.cache[self._hash(observation, proposed_action)]
        else:
            for violation in self.violations:
                if violation.in_violation(observation, proposed_action):
                    self.cache[self._hash(observation, proposed_action)] = True
                    return True
            self.cache[self._hash(observation, proposed_action)] = False
            return False

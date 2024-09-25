from collections import deque
from typing import List


class Window:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._window: deque[float] = deque(maxlen=max_size)

    def push(self, item: float) -> None:
        if len(self._window) == self.max_size:
            self._window.popleft()
        self._window.append(item)

    def get_items(self) -> List[float]:
        return list(self._window)

    def __len__(self) -> int:
        return len(self._window)

    def __str__(self) -> str:
        return str(list(self._window))

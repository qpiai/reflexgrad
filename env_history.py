"""Environment History - Already quite general, minor updates"""



from typing import List, Dict



class EnvironmentHistory:

    def __init__(self, base_query: str, start_info: str, memory: List[str], history: List[Dict[str, str]] = []) -> None:

        # Support both static and dynamic prompts

        if base_query:

            # Legacy support

            self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'

        else:

            # For dynamic prompting, start_info contains the full prompt

            self._cur_query: str = start_info

            

        self._history: List[Dict[str, str]] = history

        self._last_action: str = ''

        self._is_exhausted: bool = False

        self._interaction_count: int = 0



    def add(self, label: str, value: str) -> None:

        assert label in ['action', 'observation', 'human_edit']

        self._history += [{

            'label': label,

            'value': value,

        }]

        

        if label == 'action':

            self._interaction_count += 1

            # Check for exhaustion (same action repeated)

            if value == self._last_action:

                self._is_exhausted = True

            else:

                self._last_action = value



    def check_is_exhausted(self) -> bool:

        # Also check if too many interactions without progress

        return self._is_exhausted or self._interaction_count > 100



    def reset(self) -> None:

        self._history = []

        self._interaction_count = 0

        self._is_exhausted = False



    def get_interaction_count(self) -> int:

        return self._interaction_count



    def __str__(self) -> str:

        s: str = self._cur_query + '\n'

        for i, item in enumerate(self._history):

            if item['label'] == 'action':

                s += f'> {item["value"]}'

            elif item['label'] == 'observation':

                s += item['value']

            elif item['label'] == 'human_edit':

                s += f'[human edit]: {item["value"]}'

            if i != len(self._history) - 1:

                s += '\n'

        return s



def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:

    """Original function for backward compatibility"""

    query = base_query



    # add memory if it exists

    if len(memory) > 0:

        query += '\n\nYour memory for the task below:'

        for i, m in enumerate(memory):

            query += f'\nTrial {i}:\n{m.strip()}'

    query += f"\nHere is the task:\n{start_info}"

    return query
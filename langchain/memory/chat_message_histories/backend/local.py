from typing import List,  Dict
import time
from os import PathLike, path

import json
import threading

from langchain.prompts.chat import (
    HumanMessage, 
    AIMessage, 
    SystemMessage
)

from langchain.schema import (
    BaseMessageHistoryBackend, 
    BaseMessage
)

MESSAGE_CLS_MAP = {
    'ai': AIMessage,
    'human': HumanMessage, 
    'system': SystemMessage,
}
# Translate BaseMessage to Dict
def from_base_message_to_dict(m: BaseMessage) -> Dict: 
    return {'type': m.type, 'content':m.content}

# Translate Dict to BaseMessage
def from_dict_to_base_message(m: Dict) -> BaseMessage:
    cls = MESSAGE_CLS_MAP[m['type'].lower()] 
    return cls(content=m['content'])

# Backend to store message history in a JSON file  
class FileBackend(BaseMessageHistoryBackend):
    path:PathLike
    write_buffer:List[BaseMessage]
    buffer_size:int
    flush_period:int
    _periodic_saver_shutdown: bool
    
    def __init__(self, path: PathLike, buffer_size:int = 100, flush_period:int=30) -> None:
        super().__init__()
        self.path = path  # path to JSON file
        self.write_buffer = []
        self.buffer_size = buffer_size
        self.flush_period = flush_period
        self._periodic_saver_shutdown = False
        threading.Thread(target=self._save_periodically, daemon=True).start()

    # Retrieve last n messages from history 
    async def retrieve_history(self, n_history: int=10) -> List[BaseMessage]:
        try:
            with open(self.path, mode='r') as fp:  
                data = json.load(fp)  
                if 'history' not in data: # check if history exists
                    return []
            return [from_dict_to_base_message(log) for log in data['history'][-n_history:]]
        except Exception as e:
            return []
    
    def close(self):
        self._periodic_saver_shutdown = True
        if self.write_buffer:
            self._save_and_clear_buffer()

    # Append new messages to history
    async def push_history(self, messages: List[BaseMessage]):
        self.write_buffer += messages
        if len(self.write_buffer) > self.buffer_size:
            self._save_and_clear_buffer()
            
    def _save_periodically(self):
        while not self._periodic_saver_shutdown:
            time.sleep(self.flush_period)
            if self.write_buffer:
                self._save_and_clear_buffer()
        
    def _save_and_clear_buffer(self):
        
        if not path.exists(self.path):
            with open(self.path, 'w') as fp:
                fp.write("")
        with open(self.path, mode='r') as fp:  
            try:  
                data = json.load(fp) # load existing data from file
            except json.JSONDecodeError as e:  
                data = dict() # start with an empty dict
            except:
                raise                
            if 'history' not in data: # check if history exists
                data['history'] = []  
            data['history'] += [from_base_message_to_dict(m) for m in self.write_buffer]
            self.write_buffer.clear()
        with open(self.path, mode='w') as fp:  
            json.dump(data, fp) # dump updated data to file
            
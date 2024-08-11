import unittest
from queue import Queue
from threading import Event
from service.llm_adapter import LLM_SSE_Adapter 

class TestLLM_SSE_Adapter(unittest.TestCase):

    def test_init(self):
        """Test initialization of LLM_SSE_Adapter."""
        adapter = LLM_SSE_Adapter()
        self.assertIsInstance(adapter.msg_queue, Queue)
        self.assertFalse(adapter.finish)
        self.assertIsInstance(adapter.signal, Event)

if __name__ == '__main__':
    unittest.main()
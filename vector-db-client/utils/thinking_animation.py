import threading
import time
import sys

class ThinkingAnimation:
    """A utility class to display an animated 'Thinking...' message while waiting for LLM responses."""
    
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.dots = ["", ".", "..", "..."]
        self.current_dot_index = 0
        self.running = False
        self.thread = None
    
    def _animate(self):
        """Internal method to handle the animation loop."""
        while self.running:
            # Clear the current line and print the message with dots
            sys.stdout.write(f"\r{self.message}{self.dots[self.current_dot_index]}")
            sys.stdout.flush()
            
            # Move to next dot state
            self.current_dot_index = (self.current_dot_index + 1) % len(self.dots)
            
            # Wait before next animation frame
            time.sleep(0.5)
    
    def start(self):
        """Start the thinking animation."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the thinking animation and clear the line."""
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            
            # Clear the line by overwriting with spaces and return to beginning
            sys.stdout.write(f"\r{' ' * (len(self.message) + 3)}\r")
            sys.stdout.flush()
    
    def __enter__(self):
        """Context manager entry - start animation."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop animation."""
        self.stop()

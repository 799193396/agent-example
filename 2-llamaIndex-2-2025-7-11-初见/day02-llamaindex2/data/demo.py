# demo.py
def add(a, b):
    """Add two numbers."""
    return a + b


class Calculator:
    def multiply(self, x, y):
        return x * y

    @staticmethod
    def divide(x, y):
        """Divide x by y."""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y

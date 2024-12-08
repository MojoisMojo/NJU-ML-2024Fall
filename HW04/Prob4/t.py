class A:
    def __init__(self):
        self.val = 'a'
    
    def hh(self):
        print(f"In A, hh{self.val}")

class B(A):
    def __init__(self):
        super().__init__()
        self.val = 'b'
    
    def hh(self):
        super().hh()
        print(f"In B, hh{self.val}")

b = B()
b.hh()
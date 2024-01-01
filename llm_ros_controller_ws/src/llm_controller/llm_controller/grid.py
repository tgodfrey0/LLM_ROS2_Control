from enum import Enum 

class Grid():
  class Heading(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
  
  def __init__(self, loc: str, heading: Heading, width: int, height: int):
    self.col = loc[0].upper()
    self.row = int(loc[1])
    self.max_height = height
    self.max_width = width
    self.heading = heading
    
  def __repr__(self) -> str:
    return f"{self.col}{self.row}"
  
  def _check_bound_min_row(self) -> bool:
    b = self.row < 0
    
    if(b):
      print("Row clipped at lower bound")
    
    return b
  
  def _check_bound_max_row(self) -> bool:
    b = self.row >= self.max_height
    
    if(b):
      print("Row clipped at upper bound")
    
    return b
  
  def _check_bound_min_col(self) -> bool:
    b = (ord(self.col)-ord('A')) < 0
    
    if(b):
      print("Column clipped at lower bound")
    
    return b
  
  def _check_bound_max_col(self) -> bool:
    b = (ord(self.col)-ord('A')) >= self.max_width
    
    if(b):
      print("Column clipped at upper bound")
    
    return b
  
  def _bound_loc(self):
    self.row = 0 if self._check_bound_min_row() else self.row
    self.row = (self.max_height-1) if self._check_bound_max_row() else self.row
    self.col = 'A' if self._check_bound_min_col() else self.col
    self.col = chr((self.max_width-1) + ord('A')) if self._check_bound_max_col() else self.col
    
  def _finish_move(self):
    self._bound_loc()
    print(f"Current grid location: {self}")
    print(f"Current heading: {self.heading.name}")
  
  def forwards(self):
    match self.heading:
      case Grid.Heading.UP:
        self.row += 1
      case Grid.Heading.DOWN:
        self.row -= 1
      case Grid.Heading.LEFT:
        self.col = chr(ord(self.col)-1)
      case Grid.Heading.RIGHT:
        self.col = chr(ord(self.col)+1)
    
    self._finish_move()
  
  def backwards(self):
    match self.heading:
      case Grid.Heading.UP:
        self.row -= 1
      case Grid.Heading.DOWN:
        self.row += 1
      case Grid.Heading.LEFT:
        self.col = chr(ord(self.col)+1)
      case Grid.Heading.RIGHT:
        self.col = chr(ord(self.col)-1)

    self._finish_move()
  
  def clockwise(self):
    self.heading = Grid.Heading((self.heading.value + 1) % 4)
      
    self._finish_move()
  
  def anticlockwise(self):
    self.heading = Grid.Heading((self.heading.value - 1) % 4)
      
    self._finish_move()
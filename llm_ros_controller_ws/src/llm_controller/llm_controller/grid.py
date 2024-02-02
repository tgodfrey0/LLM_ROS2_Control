from enum import EnumMeta, Enum 

class Grid():
  class Heading(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
  
  def __init__(self, loc: str, heading: Heading, columns: int, rows: int):
    self.col = loc[0].upper()
    self.row = int(loc[1])
    self.max_height = rows
    self.max_width = columns
    self.heading = heading
    
  def __repr__(self) -> str:
    return f"{self.col}{self.row}"
  
  def _print_heading(self) -> str:
    match self.heading.value:
      case 0:
        return "Up"
      case 1:
        return "Right"
      case 2:
        return "Down"
      case 3:
        return "Left"
  
  def _check_bound_min_row(self, r) -> bool:
    b = r < 0
    
    if(b):
      print("Row clipped at lower bound")
    
    return b
  
  def _check_bound_max_row(self, r) -> bool:
    b = r >= self.max_height
    
    if(b):
      print("Row clipped at upper bound")
    
    return b
  
  def _check_bound_min_col(self, c) -> bool:
    b = (ord(c)-ord('A')) < 0
    
    if(b):
      print("Column clipped at lower bound")
    
    return b
  
  def _check_bound_max_col(self, c) -> bool:
    b = (ord(c)-ord('A')) >= self.max_width
    
    if(b):
      print("Column clipped at upper bound")
    
    return b
  
  def _bound_loc(self):
    self.row = 0 if self._check_bound_min_row(self.row) else self.row
    self.row = (self.max_height-1) if self._check_bound_max_row(self.row) else self.row
    self.col = 'A' if self._check_bound_min_col(self.col) else self.col
    self.col = chr((self.max_width-1) + ord('A')) if self._check_bound_max_col(self.col) else self.col
    
  def _finish_move(self):
    self._bound_loc()
    print(f"Current grid location: {self}")
    print(f"Current heading: {self.heading.name}")
  
  def check_forwards(self):
    r = self.row
    c = self.col

    match self.heading:
      case Grid.Heading.UP:
        r += 1
      case Grid.Heading.DOWN:
        r -= 1
      case Grid.Heading.LEFT:
        c = chr(ord(c)-1)
      case Grid.Heading.RIGHT:
        c = chr(ord(c)+1)
    
    return (self._check_bound_min_col(c)) or (self._check_bound_max_col(c)) or (self._check_bound_min_row(r)) or (self._check_bound_max_row(r))
  
  def check_backwards(self):
    r = self.row
    c = self.col

    match self.heading:
      case Grid.Heading.UP:
        r -= 1
      case Grid.Heading.DOWN:
        r += 1
      case Grid.Heading.LEFT:
        c = chr(ord(c)+1)
      case Grid.Heading.RIGHT:
        c = chr(ord(c)-1)

    return (self._check_bound_min_col(c)) or (self._check_bound_max_col(c)) or (self._check_bound_min_row(r)) or (self._check_bound_max_row(r))
  
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
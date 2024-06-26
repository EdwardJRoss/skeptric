---
categories:
- python
date: '2022-03-02T22:15:29+11:00'
image: /images/dict_to_dataclass.png
title: Dictionary to Dataclass
---

[Dataclasses](https://docs.python.org/3/library/dataclasses.html) are a really lightweight way to make classes.
When I'm programming I'll often start out with a dictionary of data and specific functions to manipulate the dictionary.
At some point it makes sense to convert it to a dataclass to package the functions with the dictionary, and enable static validation and autocompletion.

However it's not completely trivial to turn a dictionary into a dataclass.
Dictionaries are accessed by using square brackets, `d['a']`, but dataclass attributes are accessed with a dot, `d.a`.
Changing all accesses to the dictionary at once can be quite a large task.

Let's illustrate this with a simple example of representing a person as a dictionary:

```python
person = {'id': 1, 'name': 'Bob', 'age': 32}

def person_old_enough(person):
    return person['age'] >= 18
    
def person_increment_age(person):
    person['age'] = person['age'] + 1
```

We could then try to represent the person as a dataclass.
Then we'd end up with something like:

```python
from dataclasses import dataclass

@dataclass
class Person:
  id: int
  name: str
  age: int
  
  def old_enough(self) -> bool:
    return self.age >= 18
    
  def increment_age(self) -> None:
    self.age = self.age + 1
```

However if this person structure is used in lots of places across lots of files this is a very large and difficult refactoring.
Instead if we add the magic methods to get and set items to operate on the corresponding attribute we don't need to change how the object is called (as long as `del` isn't used on any fields).

```python
from dataclasses import dataclass

@dataclass
class Person:
  id: int
  name: str
  age: int
  
  def __getitem__(self, key):
    return getattr(self, key)
    
  def __setitem__(self, key, value):
    return setattr(self, key, value)

person = Person(id=1, name='Bob', age=32)

def person_old_enough(person):
    return person['age'] >= 18
    
def increment_age(person):
    person['age'] = person['age'] + 1
```

This enables us to slowly refactor the methods over time; first updating the functions to use `.` notation and then moving them into the dataclass.
Even if we don't ever get to this end state new uses of the dataclass can use the `.` notation and be statically validated by mypy to protect against typos.
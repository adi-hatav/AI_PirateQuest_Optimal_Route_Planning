### AI-PirateQuest: Optimal Route Planning

This project was an exercise as part of an introduction to artificial intelligence course.

In this project our goal was to find an optimal solution where we collect all the treasures from the islands that contain treasure and deposit them in the base while avoiding the marines.

The rules were:
- Sail Action: Move one tile vertically or horizontally.
the ship cannot move diagonally, and it cannot sail into an island.

- Collect Treasure Action: The ship can collect treasures if it is in an adjacent tile to an island with a treasure.
The ship can hold up to two treasures at once

- Deposit Treasure Action: Depositing the treasures held by the pirate ship provided that the ship is in the pirate base

- Wait Action: Choose to wait.

There can be multiple pirate ships and their starting point is at the pirate base.
In addition, there may be several Marine ships sailing on a predetermined round trip route.
If a marine ship and a pirate ship are on the same square, the treasures held by the pirate ship are confiscated.

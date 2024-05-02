# Demonstration and Visualization of the A* Algorithm a 2D Grid World
# Compatible with Python 3.8.1 and pygame 2.1.2

# Importing the Required Libraries
import numpy as np
import pygame
import math
import random

# Information for saving the animation frames
dir_name = "rrt_star_frames"
frame_number = 0

# Initializing variables defining the world and algorithm
# Can be varied as per convenience and world/algorithm specifications
WINDOW_LENGTH = 1000					# Length of the grid world along the X-axis
WINDOW_BREADTH = 1000					# Length of the grid world along the Y-axis
NODE_RADIUS = 3							# Radius of the circle displayed for each node
GOAL_RADIUS = 20						# Radius of goal reachability to ensure the algorithm has finished
EPSILON = 15							# Determines how far to place each node from its parent
REWIRING_RADIUS = 30   					# Radius to search for nodes to rewire/compare cost

# Defining different map types (based on obstacles) to perform RRT* on
# Rectangles: (left,top,width,height)
# Circles: (centre_x,centre_y,radius)
OBSTACLES = [{"rectangles":[(300,300,150,600),(700,500,250,100)],"circles":[(850,150,100)]},{"rectangles":[(700,50,50,900)],"circles":[(350,650,200),(900,300,50)]}]
OBSTACLES_CLEARANCE = [{"rectangles":[(295,295,160,610),(695,495,260,110)],"circles":[(850,150,105)]},{"rectangles":[(695,45,60,910)],"circles":[(350,650,205),(900,300,55)]}]
MAP_TYPE = 1

# Defining colour values across the RGB Scale
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (254, 254, 254)
GREEN = (0,255,0)
RED = (255,0,0)
BLUE = (0,0,255)
ORANGE = (255,164.5,0)

TARGET_REACHED = False
LDV_PROB = 0.8

# A class to define the characteristics of each grid cell (generalized to each discrete data point in a robotic configuration space)
class Node:

	def __init__(self,coord,start_node=False,target_node=False):
		self.x = coord[0]
		self.y = coord[1]
		self.parent = None
		self.children = []
		self.cost =  1e7
		self.visibility = 0.0
		self.start_node = start_node
		self.target_node = target_node
		self.best_path_cost = 1e10
		#self.best_goal_parent = None
		self.all_goal_parents = []
		if self.start_node:
			self.cost = 0

	# Colouring a node for visualization processes in pygame
	def visualize_node(self,viz_window):
		global frame_number
		colour = RED
		if self.start_node or self.target_node:
			colour = ORANGE
		pygame.draw.circle(viz_window,colour,(self.x,self.y),NODE_RADIUS,width=0)
		pygame.display.update()
		#pygame.image.save(viz_window,dir_name+"/frame"+str(frame_number)+".jpg")
		frame_number += 1

# Initializing obstacles on the map
def initialize_obstacles(viz_window):
	global frame_number
	obstacle_list = {"rectangles":[],"circles":[]}
	for rect in OBSTACLES_CLEARANCE[int(MAP_TYPE)]["rectangles"]:
		obstacle_list["rectangles"].append(pygame.Rect(rect)) 
		pygame.draw.rect(viz_window,GRAY,pygame.Rect(rect))
	for circle in OBSTACLES_CLEARANCE[int(MAP_TYPE)]["circles"]:
		obstacle_list["circles"].append(circle)
		pygame.draw.circle(viz_window,GRAY,(circle[0],circle[1]),circle[2],width=0)
	for rect in OBSTACLES[int(MAP_TYPE)]["rectangles"]:
		obstacle_list["rectangles"].append(pygame.Rect(rect)) 
		pygame.draw.rect(viz_window,BLACK,pygame.Rect(rect))
	for circle in OBSTACLES[int(MAP_TYPE)]["circles"]:
		obstacle_list["circles"].append(circle)
		pygame.draw.circle(viz_window,BLACK,(circle[0],circle[1]),circle[2],width=0)
	pygame.display.update()
	#pygame.image.save(viz_window,dir_name+"/frame"+str(frame_number)+".jpg")
	frame_number += 1
	return obstacle_list

# Checking if the new point generated collides with any obstacles in the environment
def obstacle_collision(point,obstacle_list):
	for rect in obstacle_list["rectangles"]:
		if rect.collidepoint(point):
			return True
	for circle in obstacle_list["circles"]:
		if math.sqrt((point[0]-circle[0])**2 + (point[1]-circle[1])**2) < circle[2]:
			return True
	return False

def in_obstacle(viz_window,point):
	check_point = np.array([int(point[0]),int(point[1])])
	if np.array_equal(viz_window.get_at(check_point)[:3],BLACK):
		return True
	else:
		return False

def is_corner_node(viz_window,point):
	check_point = np.array([int(point[0]),int(point[1])])
	if np.array_equal(viz_window.get_at(check_point)[:3],GRAY):
		return True
	else:
		return False

def random_sample():
	return (random.random()*WINDOW_LENGTH,random.random()*WINDOW_BREADTH)

def LDV_sample(corner_node_list):
	importances = []
	for node in corner_node_list:
		importances.append(node.visibility)
	importances = np.array(importances)
	importances = importances/importances.sum()
	LDV_sample = corner_node_list[int(np.random.choice(importances.shape[0],1,True,importances)[0])]
	theta = math.atan2(LDV_sample.y-LDV_sample.parent.y,LDV_sample.x-LDV_sample.parent.x)
	new_pos = [LDV_sample.x+EPSILON*math.cos(theta),LDV_sample.y+EPSILON*math.sin(theta)]
	new_pos[0] += np.random.normal(EPSILON/2,EPSILON/2)
	new_pos[1] += np.random.normal(EPSILON/2,EPSILON/2)
	new_pos[0] = max(min(new_pos[0],WINDOW_LENGTH-1),0)
	new_pos[1] = max(min(new_pos[1],WINDOW_BREADTH-1),0)
	return (new_pos[0],new_pos[1])

def sample_point(corner_node_list):
	if not TARGET_REACHED:
		return random_sample()
	else:
		if np.random.uniform(0,1) > LDV_PROB:
			return random_sample()
		else:
			return LDV_sample(corner_node_list)

# Recursively updating the costs of all the child nodes when a parent node gets rewired
def update_children(node):
	for child_node in node.children:
		child_node.cost = node.cost + math.sqrt((child_node.x-node.x)**2 + (child_node.y-node.y)**2)
		update_children(child_node)

# Finding the proximal parent node (as opposed to the closest node in RRT)
def find_proximal_node(new_node,node_list):
	proximal_node = None
	for node in node_list:
		dist = math.sqrt((new_node.x-node.x)**2 + (new_node.y-node.y)**2)
		if dist < REWIRING_RADIUS:
			if node.cost + dist < new_node.cost:
				proximal_node = node
				new_node.cost = node.cost + dist
				new_node.parent = proximal_node
	if proximal_node is None:
		return new_node, False
	else:
		proximal_node.children.append(new_node)
		return new_node, True

def get_node_visibility(viz_window,node,theta,step=3):
	visibility = 0.0
	current = [node.x,node.y]
	while True:
		if current[0] < 0 or current[1] < 0 or current[0] >= WINDOW_LENGTH or current[1] >= WINDOW_BREADTH or in_obstacle(viz_window,(current[0],current[1])):
			return visibility
		current[0] += step*math.cos(theta)
		current[1] += step*math.sin(theta)
		visibility += step

# Rewiring the nodes in the vicinity of the newly added node
def rewire_nodes(new_node,node_list):
	if new_node.target_node:
		return
	for node in node_list:
		dist = math.sqrt((new_node.x-node.x)**2 + (new_node.y-node.y)**2)
		if dist < REWIRING_RADIUS:
			if new_node.cost + dist < node.cost:
				node.cost = new_node.cost + dist
				if node in node.parent.children:
					node.parent.children.remove(node)
				node.parent = new_node
				update_children(node)
				new_node.children.append(node)
				theta = math.atan2(node.y-node.parent.y,node.x-node.parent.x)
				node.visibility = get_node_visibility(viz_window,node,theta)

# Adding a new node while making sure it doesn't collide with any obstacles
def add_new_node(viz_window,node_list,corner_node_list,obstacle_list):
	while True:
		#point = (random.random()*WINDOW_LENGTH,random.random()*WINDOW_BREADTH)
		point = sample_point(corner_node_list)
		nearest_node_dist = 1e7
		nearest_node = None
		for node in node_list:
			dist = math.sqrt((point[0]-node.x)**2 + (point[1]-node.y)**2)
			if dist < nearest_node_dist:
				nearest_node_dist = dist
				nearest_node = node
		theta = math.atan2(point[1]-nearest_node.y,point[0]-nearest_node.x)
		new_pos = (int(nearest_node.x+EPSILON*math.cos(theta)),int(nearest_node.y+EPSILON*math.sin(theta)))
		if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= WINDOW_LENGTH or new_pos[1] >= WINDOW_BREADTH:
			continue
		if not in_obstacle(viz_window,new_pos):
			new_node = Node(new_pos,False,False)
			new_node, success = find_proximal_node(new_node,node_list)
			if not success:
				new_node.parent = nearest_node
				nearest_node.children.append(new_node)
			node_list.append(new_node)
			theta = math.atan2(new_node.y-new_node.parent.y,new_node.x-new_node.parent.x)
			new_node.visibility = get_node_visibility(viz_window,new_node,theta)
			if is_corner_node(viz_window,(new_node.x,new_node.y)):
				corner_node_list.append(new_node)
			rewire_nodes(new_node,node_list)
			new_node.visualize_node(viz_window)
			pygame.draw.line(viz_window,BLUE,(new_node.x,new_node.y),(new_node.parent.x,new_node.parent.y))
			return new_node, node_list, corner_node_list

# Checking if the latest node added falls within the goal circle (thus completing the search)
def target_reached(node,goal):
	if math.sqrt((node.x-goal.x)**2 + (node.y-goal.y)**2) < GOAL_RADIUS:
		return True
	return False

# Highliting the final RRT path from starting to target node
def display_final_path(viz_window,goal_node):
	global frame_number
	current_node = goal_node
	while not current_node.start_node:
		pygame.draw.line(viz_window,GREEN,(current_node.x,current_node.y),(current_node.parent.x,current_node.parent.y),width=5)
		current_node = current_node.parent
	pygame.display.update()
	#pygame.image.save(viz_window,dir_name+"/frame"+str(frame_number)+".jpg")
	frame_number += 1

def display_all_paths(viz_window,goal_node):
	global frame_number
	#for parent_node in goal_node.all_goal_parents:
	#	pygame.draw.line(viz_window,RED,(goal_node.x,goal_node.y),(parent_node.x,parent_node.y),width=5)
	#	current_node = parent_node
	#	while not current_node.start_node:
	#		pygame.draw.line(viz_window,RED,(current_node.x,current_node.y),(current_node.parent.x,current_node.parent.y),width=5)
	#		current_node = current_node.parent
	pygame.draw.line(viz_window,RED,(goal_node.x,goal_node.y),(goal_node.parent.x,goal_node.parent.y),width=5)
	current_node = goal_node.parent
	while not current_node.start_node:
		pygame.draw.line(viz_window,RED,(current_node.x,current_node.y),(current_node.parent.x,current_node.parent.y),width=5)
		current_node = current_node.parent
	pygame.display.update()
	#pygame.image.save(viz_window,dir_name+"/frame"+str(frame_number)+".jpg")
	frame_number += 1

# The RRT* Algorithm
def rrt_algorithm(viz_window,start_node,goal_node,obstacle_list):
	global TARGET_REACHED
	node_list = []
	corner_node_list = []
	node_list.append(start_node)
	i = 0
	while True:
		new_node, node_list, corner_node_list = add_new_node(viz_window,node_list,corner_node_list,obstacle_list)
		if target_reached(new_node,goal_node):
			TARGET_REACHED = True
			goal_node.all_goal_parents.append(new_node)
			path_cost = (goal_node.x - new_node.x)**2 + (goal_node.y - new_node.y)**2 + new_node.cost
			if path_cost < goal_node.best_path_cost:
				goal_node.best_path_cost = path_cost
				goal_node.parent = new_node
			pygame.draw.line(viz_window,BLUE,(new_node.x,new_node.y),(new_node.parent.x,new_node.parent.y))
			if i == 0:
				node_list.append(goal_node)
				i += 1
			display_all_paths(viz_window,goal_node)
			#display_final_path(viz_window,goal_node)

# Initializing the grid world as a pygame display window,
pygame.display.set_caption('RRT* Path Finding Algorithm Visualization')
viz_window = pygame.display.set_mode((WINDOW_LENGTH,WINDOW_BREADTH))
viz_window.fill(WHITE)
pygame.display.update()

# Running RRT till completion
execute = True
start_pos, target_pos = None, None
start_node_found, target_node_found = False, False
start_node, target_node = None, None
obstacle_list = initialize_obstacles(viz_window)
while execute:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			execute = False
		elif pygame.mouse.get_pressed()[0]:
			pos = pygame.mouse.get_pos()
			if not start_node_found and pos!=target_pos:
				start_pos = pos
				start_node = Node(start_pos,True,False)
				start_node.visualize_node(viz_window)
				start_node_found = True
			elif not target_node_found and pos!=start_pos:
				target_pos = pos
				target_node = Node(target_pos,False,True)
				target_node.visualize_node(viz_window)
				target_node_found = True
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE and start_node_found and target_node_found:
				rrt_algorithm(viz_window,start_node,target_node,obstacle_list)
			if event.key == pygame.K_c:
				start_node = None
				goal_node_node = None
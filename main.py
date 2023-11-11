import numpy as np
import random
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import pandas as pd


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class GeneticAlgorithm:
    def __init__(self, points, population_size, generations):
        self.points = points
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_fitness = float('inf')
        self.best_solution = []

    def initialize_population(self):
        for _ in range(self.population_size):
            solution = self.randomize_solution()
            self.population.append(solution)

    def randomize_solution(self):
        solution = [0]
        rand_solution = list(range(1,len(self.points)))
        random.shuffle(rand_solution)
        for sol in rand_solution:
            solution.append(sol)
        return solution

    def calculate_fitness(self, solution):
        fitness = 0
        for i in range(len(solution) - 1):
            current_point = self.points[solution[i]]
            next_point = self.points[solution[i+1]]
            fitness += self.calculate_distance(current_point, next_point)
        return fitness

    def calculate_distance(self, point1, point2):
        return np.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

    def crossover(self, parent1, parent2):
        child = [-1] * len(parent1)
        child[0] = parent1[0]
        start_index = random.randint(1, len(parent1)-2)
        end_index = random.randint(start_index+1, len(parent1)-1)
        for i in range(start_index, end_index+1):
            child[i] = parent1[i]
        j = 0
        for i in range(len(parent2)):
            if child.count(parent2[i]) == 0:
                while child[j] != -1:
                    j += 1
                child[j] = parent2[i]
        return child

    def mutate(self, solution):
        index1 = random.randint(1, len(solution)-1)
        index2 = random.randint(1, len(solution)-1)
        solution[index1], solution[index2] = solution[index2], solution[index1]
        return solution

    def evolve(self):
        self.initialize_population()
        for _ in range(self.generations):
            fitness_scores = []
            for solution in self.population:
                fitness = self.calculate_fitness(solution)
                fitness_scores.append((solution, fitness))
            fitness_scores.sort(key=lambda x: x[1])
            if fitness_scores[0][1] < self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_solution = fitness_scores[0][0]
            next_generation = [self.best_solution]
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.choices(fitness_scores[:int(self.population_size/2)], k=2)
                child = self.crossover(parent1[0], parent2[0])
                if random.random() < 0.05:
                    child = self.mutate(child)
                next_generation.append(child)
            self.population = next_generation


    def plot_best_solution(self):
        x = [self.points[i].x for i in self.best_solution]
        y = [self.points[i].y for i in self.best_solution]
        coordinates = list(zip(x, y))
        # plt.plot(x, y, 'o-')
        # plt.show()
        return coordinates,self.best_solution


offices=[]
tasks_list=[]
workers_list=[]
with open("tasks.txt", "r") as file:
    n = int(file.readline())
    for i in range(n):
        line = file.readline().replace("\n","")
        task_id,priority,latitude,longitude = line.split(" ")
        tasks_list.append(
            {"id": int(task_id), "priority": int(priority), "latitude": float(latitude),"longitude":float(longitude),"office_destination":0})
with open("workers.txt", "r") as file:
    n = int(file.readline())
    for i in range(n):
        line = file.readline().replace("\n","")
        task_id,grade,latitude,longitude = line.split(" ")
        if not offices:
            offices.append((latitude, longitude))
        new = 0
        for off in offices:
            if off!=(latitude,longitude):
               new+=1
        if new==len(offices):
            offices.append((latitude,longitude))
        workers_list.append(
            {"id": task_id, "grade": int(grade), "latitude": float(latitude),"longitude":float(longitude),"hours":9})
#priority 3 - сложное,2-среднее,1-легкое
# Сортировка по приоритету и расстоянию
for task in tasks_list:
    sum_distance = 0
    for office in offices:
        distance = geodesic((task["latitude"], task["longitude"]), office).kilometers
        sum_distance += distance
    task["office_destination"] = sum_distance

tasks_list.sort(key=lambda x: (x['priority'], x['office_destination']), reverse=True)
# 3-синьор, 2-мидл, 1-джун
workers_list.sort( key=lambda x: (x["grade"], x["hours"]), reverse=True)

# Инициализируем словарь для хранения назначенных задач на каждого работника
assigned_tasks = {worker_id["id"]: [] for worker_id in workers_list}
coordinates = {worker_id["id"]: [] for worker_id in workers_list} #координаты для каждого работника
roots = {worker_id["id"]: [] for worker_id in workers_list} #лучший путь для каждого сотрудника
# Перебор задач и распределяем их на работников
for task in tasks_list:
    # Сортировка группы работников по ближайшему расстоянию до задачи
    workers_list.sort(key=lambda x: geodesic((x["latitude"],x["longitude"]), (task["latitude"],task["longitude"])))
    workers_list.sort(reverse=True, key=lambda x:x["hours"],) # по оставшемуся времени
    if task["priority"] == 3:
        hourse = 4
    elif task["priority"] == 2:
        hourse= 2
    elif task["priority"] == 1:
        hourse= 1.5
    for worker in workers_list:
        if (task["priority"] <= worker["grade"]) and (worker["hours"] -hourse>=0):
                assigned_tasks[worker["id"]].append(task["id"])
                if not coordinates[worker["id"]]:
                    coordinates[worker["id"]].append((worker["latitude"],worker["longitude"]))
                coordinates[worker["id"]].append((task["latitude"],task["longitude"]))
                worker["hours"] -= hourse+geodesic((worker["latitude"],worker["longitude"]), (task["latitude"],task["longitude"])).kilometers/30 #30 это примерная скорость автомобиля
                break;

result=[]
for key,value in coordinates.items():
    points=[]
    task_dic=[]
    coor_list=value
    if len(coor_list)>2:
        for coor in coor_list:
            x,y=coor
            points.append(Point(x,y))
        genetic_algorithm = GeneticAlgorithm(points, population_size=50, generations=200)
        genetic_algorithm.evolve()
        location,solution = genetic_algorithm.plot_best_solution()
        task_ids= assigned_tasks[key]
        for i in range(1,len(solution)):
            task = {
                "ID таска": task_ids[solution[i]-1],
                "широта": location[i][0],
                "долгота": location[i][1]
                }
            task_dic.append(task)
        data_dict = {
            "ID работника": key,
            "таски": task_dic
        }
        roots[key].append(location)

    else:
        roots[key].append(value)
        task_ids = assigned_tasks[key]
        task_dic.append({
                "ID таска": task_ids[0],
                "широта": value[1][0],
                "долгота": value[1][1]
                })
        #task_dic.append(task)
        data_dict = {
            "ID работника": key,
            "таски": task_dic
        }
    result.append(data_dict)

# root=pd.DataFrame(roots)
# root.to_excel("./roots.xlsx", index=False)
with open("output.txt", "w") as file:
    # Записываем количество входных наборов данных
    file.write(str(len(result)) + "\n")
    # Записываем данные для каждого входного набора
    for entry in result:
        file.write(f"{entry['ID работника']} {len(entry['таски'])}\n")
        for task in entry['таски']:
            file.write(f"{task['ID таска']} {task['широта']} {task['долгота']}\n")

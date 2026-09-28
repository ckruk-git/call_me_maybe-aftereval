import parser

p = parser.Parser()
p.parse_file("maps/easy/01_linear_path.txt")
adjacency_list = {}
adjacency_list[p.start_hub.name] = []
adjacency_list[p.end_hub.name] = []
for name in p.hubs:
    adjacency_list[name] = []
for conn in p.connections:
    adjacency_list[conn.a].append(conn.b)
    adjacency_list[conn.b].append(conn.a)


for name, neighbors in adjacency_list.items():
    print(f"{name} -> {neighbors}")

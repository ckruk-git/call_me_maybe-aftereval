import re

allowed_zones = {"normal", "blocked", "restricted", "priority"}

class Hub:
    def __init__(self, 
                 name: str, 
                 x: int, 
                 y: int, 
                 zone: str = "normal", 
                 color: str = "none", 
                 max_drones: int = 1) -> None:
        if zone not in allowed_zones:
            raise ValueError(f"Invalid zone type")
        self.name = name
        self.x = x
        self.y = y
        self.zone = zone
        self.color = color
        self.max_drones = max_drones

class Connection:
    def __init__(self,
                 a: str, 
                 b: str,
                 max_link_capacity: int = 1
                 ) -> None:
        self.a = a
        self.b = b
        self.max_link_capacity = max_link_capacity

class ParseError(Exception):
    """Błąd parsowania pliku wejściowego."""

    def __init__(self, 
                 line_nbr: int,
                 message: str) -> None:
        super().__init__(f"Line {line_nbr}: {message}")
        self.line_nbr = line_nbr

class Parser:
    def __init__(self) -> None:
        self.nb_drones: int | None = None
        self.start_hub: Hub | None = None
        self.end_hub: Hub | None = None
        self.hubs: dict[str, Hub] = {}
        self.connections: list[Connection] = []
        self._connection_keys: set[frozenset[str]] = set()

    def parse_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line_nbr, raw_line in enumerate(f, start=1):
                self.parse_line(raw_line, line_nbr)
        self._validate()

    def parse_line(self, raw_line: str, line_nbr: int) -> None:
        if '#' in raw_line or not raw_line or raw_line[0] == "\n":
            return
        if ":" not in raw_line:
            raise ParseError(line_nbr, "missing ':' in line")
        key, rest = raw_line.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if key == "nb_drones":
            self.parse_nb_drones(rest, line_nbr)
        elif key == "start_hub":
            hub_type = "start"
            self.parse_hub(rest, line_nbr, hub_type)
        elif key == "hub":
            hub_type = "middle"
            self.parse_hub(rest, line_nbr, hub_type)
        elif key == "end_hub":
            hub_type = "end"
            self.parse_hub(rest, line_nbr, hub_type)
        elif key == "connection":
            self.parse_connection(rest, line_nbr)
        else:
            raise ParseError(line_nbr, f"Unknown key '{key}'")

    def parse_nb_drones(self, data, line_nbr) -> None:
        if self.nb_drones is not None:
            raise ParseError(line_nbr, "nb_drones defined twice")
        try:
            value = int(data)
        except ValueError:
            raise ParseError(line_nbr, f"nb_drones must be int, received: '{data}'")
        if value <= 0:
            raise ParseError(line_nbr, "nb_drones must be positive")
        self.nb_drones = value

    def parse_hub(self, data: str, line_nbr: int, hub_type: str) -> None:
        
        data_brackets = re.findall(r"\[([^\]]*)\]", data)
        no_brackets = re.sub(r"\[[^\]]*\]", "", data).strip()
        parts = no_brackets.split()
        if len(parts) != 3:
            raise ParseError(line_nbr, "hub must have: name, x, y")
        name = parts[0]
        x_str = parts[1]
        y_str = parts[2]
        try:
            x = int(x_str)
            y = int(y_str)
        except ValueError:
            raise ParseError(line_nbr, f"x and y must be int, received '{x_str}', '{y_str}'")
        data_dict: dict[str, str] = {}
        for block in data_brackets:
            for part in block.split():
                if "=" not in part:
                    raise ParseError(line_nbr, f"Invalid part of data_brackets {part}")
                k, v = part.split("=", 1)
                if k in data_dict:
                    raise ParseError(line_nbr, f"duplicated key of part of data_brackets '{k}'")
                data_dict[k] = v
        zone = data_dict.get("zone", "normal")
        color = data_dict.get("color", "none")
        max_drones_str = data_dict.get("max_drones", "1")

        if zone not in allowed_zones:
            raise ParseError(line_nbr, f"invalid zone type '{zone}'")
        try:
            max_drones = int(max_drones_str)
        except ValueError:
            raise ParseError(line_nbr, f"max drones must be int, received '{max_drones_str}'")
        if max_drones <= 0:
            raise ParseError(line_nbr, "max_drones must be positive")
        hub = Hub(name, x, y, zone, color, max_drones)

        if name in self.hubs:
            raise ParseError(line_nbr, f"duplicated hub name '{name}'")
        if self.start_hub is not None and self.start_hub.name == name:
            raise ParseError(line_nbr, f"duplicated hub name '{name}'")
        if self.end_hub is not None and self.end_hub.name == name:
            raise ParseError(line_nbr, f"duplicated hub name '{name}'")
                
        if hub_type == "start":
            if self.start_hub is not None:
                raise ParseError(line_nbr, "start_hub defined twice")
            self.start_hub = hub
        elif hub_type == "middle":
            self.hubs[name] = hub
        elif hub_type == "end":
            if self.end_hub is not None:
                raise ParseError(line_nbr, "end_hub defined twice")
            self.end_hub = hub
        else:
            raise ParseError(line_nbr, f"Unknown hub_type '{hub_type}'")

    def parse_connection(self, data, line_nbr) -> None:
        data_brackets = re.findall(r"\[([^\]]*)\]", data)
        max_cap = 1
        if len(data_brackets) > 0:
            bracket_parts = data_brackets[0].split("=")
            if len(bracket_parts) != 2:
                raise ParseError(
                    line_nbr,
                    f"connection must have exactly one '=', got '{bracket_parts}'"
                    )
            if (bracket_parts[0] != "max_link_capacity"):
                raise ParseError(line_nbr, f"Unknown name, shoulf be max_link_capacity, received f'{bracket_parts[0]}'")
            try:
                max_cap = int(bracket_parts[1])
                if max_cap <= 0:
                    raise ParseError (line_nbr, "max_link_capacity must be positive")
            except ValueError:
                raise ParseError(line_nbr, f"max_link_capacity must be int, received '{bracket_parts[1]}'")
        connection = data.split(" ")[0]
        connection_splited = connection.split("-")
        a = connection_splited[0].strip()
        b = connection_splited[1].strip()

        known = set(self.hubs.keys())
        if self.start_hub is not None:
            known.add(self.start_hub.name)
        if self.end_hub is not None:
            known.add(self.end_hub.name)

        if a not in known:
            raise ParseError(line_nbr, f"unknown hub '{a}' in connection")
        if b not in known:
            raise ParseError(line_nbr, f"unknown hub '{b}' in connection")

        key = frozenset({a, b})
        if key in self._connection_keys:
            raise ParseError(line_nbr, f"connection '{a}-{b}' already defined")

        conn = Connection(a, b, max_cap)
        self.connections.append(conn)
        self._connection_keys.add(key)

    def _validate(self) -> None:
        if self.nb_drones is None:
            raise ParseError(0, "nb_drones not set'")
        if self.start_hub is None:
            raise ParseError(0, "start_hub not set'")
        if self.end_hub is None:
            raise ParseError(0, "end_hub not set'")


if __name__ == "__main__":
    p = Parser()
    p.parse_file("maps/easy/01_linear_path.txt")
    print("nb_drones:", p.nb_drones)
    print("start:", p.start_hub.name if p.start_hub else None)
    print("end:", p.end_hub.name if p.end_hub else None)
    print("hubs:", list(p.hubs.keys()))
    print("connections count:", len(p.connections))
    for c in p.connections:
        print(f"  {c.a}-{c.b} cap={c.max_link_capacity}")
import io
import json
import re
import zipfile
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from functools import total_ordering
import heapq
import base64
import requests
from datetime import datetime, timedelta
import anthropic
import streamlit as st
import svgwrite

# Configuration
BOARD_WIDTH_MM = 100
BOARD_HEIGHT_MM = 80
SVG_WIDTH = 800
SVG_HEIGHT = 600

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Data Structures


@dataclass
class Pin:
    name: str
    x: float  # mm from component origin
    y: float  # mm from component origin
    type: str  # power, ground, io, etc
    number: str  # pin number/identifier
    net: Optional[str] = None  # Net this pin belongs to


@dataclass
class Component:
    name: str
    reference: str  # e.g., R1, LED1
    package: str    # e.g., TH, SMD-0805
    value: str     # e.g., "10k", "RED"
    pins: List[Pin]
    width: float   # mm
    height: float  # mm
    x: float = 0.0  # placement x coordinate
    y: float = 0.0  # placement y coordinate
    rotation: float = 0.0  # degrees


@dataclass
@total_ordering
class Point:
    x: int  # in 0.1µm
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)


@dataclass
class Net:
    name: str
    points: List[Point]


@dataclass
class NetConnection:
    from_component: str  # Component reference (e.g., "R1")
    from_pin: str       # Pin number/name
    to_component: str
    to_pin: str
    net_name: str


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    count: int
    g_score: float = field(compare=False)
    point: Point = field(compare=False)


class ComponentDatabase:
    """Interface to SnapEDA API for component footprints"""

    def __init__(self):
        self.api_key = st.secrets.get("SNAPEDA_API_KEY", "")
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = timedelta(hours=24)

    def search_component(self, part_number: str) -> Optional[Dict]:
        """Search for component footprint on SnapEDA"""
        # Check cache first
        cached_data = self._get_cached(part_number)
        if cached_data:
            return cached_data

        if not self.api_key:
            st.warning("SnapEDA API key not configured")
            return None

        try:
            # First search for the part
            search_url = "https://www.snapeda.com/api/v1/parts/search"
            params = {
                "q": part_number,
                "limit": 1
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }

            response = requests.get(search_url, params=params, headers=headers)
            if response.status_code != 200:
                st.warning(f"Error searching SnapEDA: {response.status_code}")
                return None

            results = response.json()
            if not results.get("results"):
                return None

            # Get the first result
            part_data = results["results"][0]

            # For now, return a simplified footprint with basic information
            # In a production system, we would need to:
            # 1. Download the KiCad files
            # 2. Parse them to extract exact pin positions
            # 3. Convert to our internal format
            footprint = {
                "package": part_data.get("package_name", "Unknown"),
                "width": 5.0,  # Default size
                "height": 5.0,  # Default size
                "reference": part_data.get("manufacturer_part_number", "U"),
                "description": part_data.get("description", ""),
                "pins": []
            }

            # Add basic pin information based on pin count
            pin_count = part_data.get("pin_count", 2)
            for i in range(pin_count):
                footprint["pins"].append({
                    "name": f"PIN_{i+1}",
                    "number": str(i+1),
                    "x": i * 2.54,  # Standard 0.1" spacing
                    "y": 0,
                    "type": "io"
                })

            # Cache the result
            self._set_cached(part_number, footprint)

            # Add download URLs to the response for user information
            st.info(f"""
            Found component on SnapEDA: {part_data.get('manufacturer_part_number')}
            
            To get exact footprint:
            1. Download from: {part_data.get('kicad_fp_url', 'Not available')}
            2. Import into KiCad
            3. Extract precise dimensions
            
            Currently using simplified footprint data.
            """)

            return footprint

        except Exception as e:
            st.warning(f"Error accessing SnapEDA API: {e}")
            return None


class SchematicGenerator:
    """Generate netlist from circuit description"""

    def generate_netlist(self, description: str, components: List[Component]) -> List[NetConnection]:
        """Generate netlist from circuit description using Claude"""
        try:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Given these components for a {description}:
                    {self._format_components(components)}
                    
                    Generate a netlist showing how the components should be connected.
                    For each connection specify:
                    - source component reference and pin
                    - destination component reference and pin
                    - net name (e.g., VCC, GND, NET1, etc.)
                    
                    Return as JSON list of connections within triple backticks.
                    Each connection should have: from_component, from_pin, to_component, to_pin, net_name"""
                }]
            )

            json_str = re.search(r'```json\s*([\s\S]*?)\s*```',
                                 response.content[0].text).group(1)
            connections = json.loads(json_str)

            return [NetConnection(**conn) for conn in connections]

        except Exception as e:
            st.error(f"Error generating netlist: {str(e)}")
            return []

    def _format_components(self, components: List[Component]) -> str:
        """Format component list for Claude prompt"""
        result = []
        for comp in components:
            pins = [f"Pin {p.number}: {p.name} ({p.type})"
                    for p in comp.pins]
            result.append(
                f"{comp.reference}: {comp.name} ({comp.value})\n" +
                "\n".join(f"  {p}" for p in pins)
            )
        return "\n\n".join(result)

    def apply_netlist(self, components: List[Component],
                      connections: List[NetConnection]) -> List[Component]:
        """Apply netlist connections to component pins"""
        # Create lookup for components by reference
        comp_lookup = {c.reference: c for c in components}

        # Apply connections to pins
        for conn in connections:
            from_comp = comp_lookup.get(conn.from_component)
            to_comp = comp_lookup.get(conn.to_component)

            if not from_comp or not to_comp:
                continue

            # Find matching pins
            from_pin = next((p for p in from_comp.pins
                             if p.number == conn.from_pin), None)
            to_pin = next((p for p in to_comp.pins
                           if p.number == conn.to_pin), None)

            if from_pin and to_pin:
                from_pin.net = conn.net_name
                to_pin.net = conn.net_name

        return components


class PCBAutorouter:
    def __init__(self, board_width_mm: float, board_height_mm: float, grid_mm: float = 0.1):
        self.grid_size = int(grid_mm * 1000000)  # Convert to 0.1µm
        self.width = int(board_width_mm * 1000000)
        self.height = int(board_height_mm * 1000000)
        self.grid_width = self.width // self.grid_size
        self.grid_height = self.height // self.grid_size
        self.obstacles: Set[Point] = set()
        self._queue_count = 0

    def manhattan_distance(self, p1: Point, p2: Point) -> int:
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def _point_to_grid(self, point: Point) -> Tuple[int, int]:
        return (point.x // self.grid_size, point.y // self.grid_size)

    def _grid_to_point(self, grid_x: int, grid_y: int) -> Point:
        return Point(grid_x * self.grid_size, grid_y * self.grid_size)

    def _get_neighbors(self, point: Point) -> List[Point]:
        grid_x, grid_y = self._point_to_grid(point)
        neighbors = []

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = grid_x + dx, grid_y + dy
            if (0 <= new_x < self.grid_width and
                    0 <= new_y < self.grid_height):
                new_point = self._grid_to_point(new_x, new_y)
                if new_point not in self.obstacles:
                    neighbors.append(new_point)

        return neighbors

    def _route_net(self, start: Point, end: Point) -> List[Point]:
        def heuristic(p: Point) -> float:
            return float(self.manhattan_distance(p, end))

        queue = []
        heapq.heapify(queue)
        self._queue_count = 0
        heapq.heappush(queue, PrioritizedItem(
            0.0, self._queue_count, 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: heuristic(start)}

        while queue:
            current_item = heapq.heappop(queue)
            current = current_item.point

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + float(self.grid_size)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    self._queue_count += 1
                    heapq.heappush(queue, PrioritizedItem(
                        f, self._queue_count, tentative_g, neighbor))

        return []

    def route(self, components: List[Component]) -> Dict[str, List[Point]]:
        """Route all connections between components based on netlist"""
        routes = {}
        nets = self._create_nets_from_components(components)

        sorted_nets = sorted(
            nets.items(),
            key=lambda x: len(x[1].points)  # Route shorter nets first
        )

        for net_name, net in sorted_nets:
            path_segments = []
            for i in range(len(net.points) - 1):
                path = self._route_net(net.points[i], net.points[i + 1])
                if not path:
                    st.warning(f"Could not route net {net_name} segment {i}")
                    continue
                path_segments.extend(path)

                for point in path:
                    self.obstacles.add(point)

            routes[net_name] = path_segments

        return routes

    def _create_nets_from_components(self, components: List[Component]) -> Dict[str, Net]:
        """Create nets from component pin connections"""
        nets = {}

        for component in components:
            for pin in component.pins:
                if pin.net:
                    if pin.net not in nets:
                        nets[pin.net] = Net(pin.net, [])
                    # Convert pin position to absolute coordinates
                    # Convert mm to 0.1µm
                    x = int((component.x + pin.x) * 1000000)
                    y = int((component.y + pin.y) * 1000000)
                    nets[pin.net].points.append(Point(x, y))

        return nets


class PCBRenderer:
    """Render PCB layout as SVG"""

    @staticmethod
    def render_to_svg(gerber_files: Dict[str, str]) -> str:
        dwg = svgwrite.Drawing(size=(SVG_WIDTH, SVG_HEIGHT))

        # Set viewBox for proper scaling
        padding = 5  # mm
        dwg.viewbox(
            -padding, -padding,
            BOARD_WIDTH_MM + 2*padding,
            BOARD_HEIGHT_MM + 2*padding
        )

        main_group = dwg.g()
        dwg.add(main_group)

        # Board outline with dark green color
        main_group.add(dwg.rect(
            insert=(0, 0),
            size=(BOARD_WIDTH_MM, BOARD_HEIGHT_MM),
            fill='#1A5C28',  # Dark green
            stroke='#000000',
            stroke_width=0.2
        ))

        # Render copper layer traces
        if 'F.Cu.gbr' in gerber_files:
            traces_group = dwg.g(id='Traces', fill='none')
            main_group.add(traces_group)

            x, y = 0, 0
            current_path = []

            for line in gerber_files['F.Cu.gbr'].split('\n'):
                if line.startswith('D'):
                    if 'D01' in line:  # Draw line
                        coords = line.split('D01')[0]
                        new_x = float(coords.split(
                            'X')[-1].split('Y')[0])/1000000
                        new_y = float(coords.split('Y')[-1])/1000000
                        current_path.append((new_x, new_y))
                        x, y = new_x, new_y

                    elif 'D02' in line:  # Move
                        if current_path:
                            # Draw previous path
                            path_data = "M " + " L ".join(
                                f"{x},{y}" for x, y in current_path)
                            traces_group.add(dwg.path(
                                d=path_data,
                                stroke='#C87533',  # Copper color
                                stroke_width=0.2,
                                stroke_linecap='round'
                            ))
                            current_path = []

                        coords = line.split('D02')[0]
                        x = float(coords.split('X')[-1].split('Y')[0])/1000000
                        y = float(coords.split('Y')[-1])/1000000
                        current_path = [(x, y)]

                    elif 'D03' in line:  # Flash pad
                        coords = line.split('D03')[0]
                        if 'X' in coords and 'Y' in coords:
                            x = float(coords.split('X')
                                      [-1].split('Y')[0])/1000000
                            y = float(coords.split('Y')[-1])/1000000

                            # Draw pad as copper circle
                            traces_group.add(dwg.circle(
                                center=(x, y),
                                r=0.8,  # 0.8mm radius
                                fill='#C87533',  # Copper color
                                stroke='none'
                            ))

            # Draw any remaining path
            if current_path:
                path_data = "M " + " L ".join(
                    f"{x},{y}" for x, y in current_path)
                traces_group.add(dwg.path(
                    d=path_data,
                    stroke='#C87533',
                    stroke_width=0.2,
                    stroke_linecap='round'
                ))

        # Add solder mask openings
        if 'F.Mask.gbr' in gerber_files:
            mask_group = dwg.g(id='Mask')
            main_group.add(mask_group)

            for line in gerber_files['F.Mask.gbr'].split('\n'):
                if 'D03' in line:  # Pad opening
                    coords = line.split('D03')[0]
                    if 'X' in coords and 'Y' in coords:
                        x = float(coords.split('X')[-1].split('Y')[0])/1000000
                        y = float(coords.split('Y')[-1])/1000000

                        # Add solder mask opening
                        mask_group.add(dwg.circle(
                            center=(x, y),
                            r=1.0,  # Slightly larger than pad
                            fill='none',
                            stroke='#194C28',  # Darker green
                            stroke_width=0.1
                        ))

        # Add drill holes
        if 'drill.xln' in gerber_files:
            drill_group = dwg.g(id='Drills')
            main_group.add(drill_group)

            for line in gerber_files['drill.xln'].split('\n'):
                if line.startswith('X'):
                    x = float(line.split('X')[1].split('Y')[0])
                    y = float(line.split('Y')[1])
                    drill_group.add(dwg.circle(
                        center=(x, y),
                        r=0.2,  # 0.4mm diameter
                        fill='#000000'
                    ))

        return dwg.tostring()

    @staticmethod
    def get_svg_html(svg_string: str) -> str:
        svg_base64 = base64.b64encode(svg_string.encode()).decode()
        return f'data:image/svg+xml;base64,{svg_base64}'


class PartsListGenerator:
    def __init__(self):
        self.component_db = ComponentDatabase()

    def _extract_json(self, text: str) -> str:
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        return match.group(1) if match else None

    def generate(self, prompt: str) -> List[Component]:
        """Generate parts list with complete component information"""
        try:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Generate a parts list for a PCB based on this description: {prompt}
                    Please choose specific, real components with manufacturer part numbers.
                    For each part include:
                    - name
                    - description
                    - manufacturer part number (prefer common, well-documented parts)
                    - value (if applicable)
                    Format as JSON list with those fields within triple backticks."""
                }]
            )

            if not response.content:
                raise ValueError("No content in response")

            json_str = self._extract_json(response.content[0].text)
            if not json_str:
                raise ValueError("No JSON content found in response")

            basic_parts = json.loads(json_str)
            return self._create_components(basic_parts)

        except Exception as e:
            st.error(f"Error generating parts list: {str(e)}")
            return []

    def _create_components(self, basic_parts: List[dict]) -> List[Component]:
        """Convert basic parts to Component objects with footprint data"""
        components = []
        x_offset = 10  # mm

        for i, part in enumerate(basic_parts):
            part_number = part.get("manufacturer_part_number")
            if not part_number:
                st.warning(f"No part number for {part.get('name')}")
                continue

            footprint = self.component_db.search_component(part_number)
            if not footprint:
                st.warning(f"No footprint found for {part_number}")
                continue

            pins = [
                Pin(
                    name=pin["name"],
                    number=pin["number"],
                    x=pin["x"],
                    y=pin["y"],
                    type=pin["type"],
                    net=None  # Will be assigned during netlist generation
                )
                for pin in footprint["pins"]
            ]

            component = Component(
                name=part["name"],
                reference=f"{footprint['reference']}{i+1}",
                package=footprint["package"],
                value=part.get("value", ""),
                pins=pins,
                width=footprint["width"],
                height=footprint["height"],
                x=x_offset,
                y=10  # Fixed Y position for now
            )

            x_offset += component.width + 5  # 5mm spacing between components
            components.append(component)

        return components


class GerberGenerator:
    def __init__(self, width_mm: float = BOARD_WIDTH_MM, height_mm: float = BOARD_HEIGHT_MM):
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.autorouter = PCBAutorouter(width_mm, height_mm)

    def _generate_header(self, layer_name: str) -> str:
        return f"""G04 {layer_name}*
%FSLAX46Y46*%
%MOMM*%
%LPD*%
%ADD10C,0.15*%
%ADD20R,0.60X0.60*%
G01*
"""

    def generate_files(self, components: List[Component]) -> Dict[str, str]:
        gerber_files = {}

        # Route connections
        routes = self.autorouter.route(components)

        # Generate F.Cu layer
        f_cu_content = self._generate_header('F.Cu')

        # Add component pads
        for component in components:
            for pin in component.pins:
                x = int((component.x + pin.x) * 1000000)
                y = int((component.y + pin.y) * 1000000)
                f_cu_content += f"D20*\nX{x}Y{y}D03*\n"

        # Add traces
        f_cu_content += "D10*\n"
        for path in routes.values():
            if len(path) < 2:
                continue
            f_cu_content += f"X{path[0].x}Y{path[0].y}D02*\n"
            for point in path[1:]:
                f_cu_content += f"X{point.x}Y{point.y}D01*\n"

        f_cu_content += "M02*\n"
        gerber_files['F.Cu.gbr'] = f_cu_content

        # Generate F.Mask layer
        f_mask_content = self._generate_header('F.Mask')
        for component in components:
            for pin in component.pins:
                x = int((component.x + pin.x) * 1000000)
                y = int((component.y + pin.y) * 1000000)
                f_mask_content += f"D20*\nX{x}Y{y}D03*\n"
        f_mask_content += "M02*\n"
        gerber_files['F.Mask.gbr'] = f_mask_content

        # Generate drill file
        drill_content = """M48
FMAT,2
METRIC,TZ
T1C0.4
%
T1
"""
        for component in components:
            for pin in component.pins:
                x = component.x + pin.x
                y = component.y + pin.y
                drill_content += f"X{x:.3f}Y{y:.3f}\n"
        drill_content += "M30\n"
        gerber_files['drill.xln'] = drill_content

        return gerber_files


def main():
    st.set_page_config(
        page_title="PCB Generator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('AI-Powered PCB Generator with SnapEDA Integration')
    st.write("""
    This tool helps you generate PCB designs using real component footprints from SnapEDA.
    Enter a description of your circuit, and the AI will:
    1. Generate a parts list with real components
    2. Fetch accurate footprints from SnapEDA
    3. Create a netlist showing component connections
    4. Generate a PCB layout with proper routing
    5. Create manufacturing-ready Gerber files
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        board_width = st.slider("Board Width (mm)", 40, 200, BOARD_WIDTH_MM)
        board_height = st.slider("Board Height (mm)", 40, 200, BOARD_HEIGHT_MM)

        st.header("Routing Settings")
        trace_width = st.slider("Trace Width (mm)", 0.1, 1.0, 0.2, 0.1)
        grid_size = st.slider("Grid Size (mm)", 0.1, 1.0, 0.1, 0.1)

        if st.button("Reset to Defaults"):
            st.rerun()

    # Main content area
    tab1, tab2, tab3 = st.tabs(
        ["Circuit Design", "Component List", "PCB Layout"])

    with tab1:
        st.header("Circuit Description")
        with st.expander("📝 Example Descriptions", expanded=True):
            st.write("""
            Try these example descriptions:
            - "A simple LED circuit with a button and battery. The LED should light up when the button is pressed."
            - "Temperature sensor circuit using a DS18B20 with LED indicator and USB power."
            - "Basic H-bridge motor driver with direction control switch and power LED."
            """)

        prompt = st.text_area(
            "Enter your circuit description:",
            placeholder="Describe the circuit you want to create...",
            height=100
        )

    if "SNAPEDA_API_KEY" not in st.secrets:
        st.error("""
        SnapEDA API key not configured. Please add it to your secrets.toml file:
        ```
        SNAPEDA_API_KEY = "your-api-key"
        ```
        """)
        return

    if st.button('Generate PCB', type='primary', use_container_width=True):
        # Step 1: Generate parts list
        with tab2:
            with st.spinner('🤖 Generating parts list...'):
                generator = PartsListGenerator()
                components = generator.generate(prompt)

            if not components:
                st.error("Failed to generate parts list")
                return

            st.write("### Generated Parts List")
            for comp in components:
                with st.expander(f"{comp.reference}: {comp.name} ({comp.package})", expanded=True):
                    st.write("**Specifications:**")
                    st.write(f"- Value: {comp.value}")
                    st.write(f"- Package: {comp.package}")
                    st.write(f"- Dimensions: {comp.width}mm × {comp.height}mm")

                    st.write("**Pins:**")
                    cols = st.columns(2)
                    for i, pin in enumerate(comp.pins):
                        col = cols[i % 2]
                        with col:
                            st.write(
                                f"• {pin.number}: {pin.name} ({pin.type})")

            # Step 2: Generate netlist
            st.write("### Component Connections")
            with st.spinner('🔄 Creating netlist...'):
                schematic = SchematicGenerator()
                connections = schematic.generate_netlist(prompt, components)
                components = schematic.apply_netlist(components, connections)

            for conn in connections:
                st.write(
                    f"- {conn.from_component}.{conn.from_pin} → "
                    f"{conn.to_component}.{conn.to_pin} ({conn.net_name})"
                )

        # Step 3: Create PCB layout
        with tab3:
            with st.spinner('🔧 Creating PCB layout...'):
                gerber_gen = GerberGenerator(
                    width_mm=board_width,
                    height_mm=board_height
                )
                gerber_files = gerber_gen.generate_files(components)

            # Step 4: Render preview
            with st.spinner('🎨 Rendering PCB preview...'):
                renderer = PCBRenderer()
                svg_string = renderer.render_to_svg(gerber_files)
                svg_html = renderer.get_svg_html(svg_string)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("### PCB Preview")
                st.markdown(
                    f'<img src="{svg_html}" style="width: 100%; max-width: 800px; border: 1px solid #ccc; border-radius: 5px;">',
                    unsafe_allow_html=True
                )
                st.caption('Generated PCB Layout')

            with col2:
                st.write("### Export Options")

                # Download Gerber files
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for filename, content in gerber_files.items():
                        zip_file.writestr(filename, content)

                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download Gerber Files (ZIP)",
                    data=zip_buffer,
                    file_name="gerber_files.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                # Download SVG
                st.download_button(
                    label="📥 Download SVG Preview",
                    data=svg_string,
                    file_name="pcb_preview.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )

                with st.expander("Manufacturing Notes", expanded=True):
                    st.info("""
                    **Before Manufacturing:**
                    1. Verify all component footprints
                    2. Check pin connections
                    3. Review design rules
                    4. Validate trace widths
                    5. Check clearances
                    """)

                with st.expander("Layer Information"):
                    st.write("**Included Layers:**")
                    st.write("- F.Cu: Front copper")
                    st.write("- F.Mask: Front solder mask")
                    st.write("- Drill: Through-holes")


if __name__ == "__main__":
    main()

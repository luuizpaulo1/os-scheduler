import os
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from itertools import count, chain
from typing import Optional, Any, TypeVar

from pydantic import BaseModel, ConfigDict
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

process_id = count(1)
disk_id = count(1)
cpu_id = count(1)

gb_in_mb = 1024

T = TypeVar("T")


def assert_is_not_none(t: Optional[T]) -> T:
    assert t is not None
    return t


class Disk:
    def __init__(self) -> None:
        self.id = next(disk_id)
        self.process: Optional[Process] = None

    def deallocate(self) -> None:
        self.process = None


class ProcessState(Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    SUSPENDED_READY = "SUSPENDED_READY"
    SUSPENDED_BLOCKED = "SUSPENDED_BLOCKED"
    EXIT = "EXIT"


class Process(BaseModel):
    id: int
    arrival: int
    phase_1_duration: int
    io_duration: int
    phase_2_duration: int
    size_in_mb: int
    disk_quantity: int
    state: ProcessState = ProcessState.NEW
    actual_queue: int = 0
    last_executed_time: int = -1
    allocated_in_memory: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def remaining_time(self) -> int:
        return self.phase_1_duration + self.phase_2_duration

    def wait_for_io(self) -> None:
        self.io_duration -= 1

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)


class ProcessQueue:
    def __init__(self) -> None:
        self.processes: list[Process] = []

    @property
    def empty(self) -> bool:
        return len(self.processes) == 0

    def push(self, process: Process) -> None:
        self.processes.append(process)

    def pop(self) -> Process:
        return self.processes.pop(0)


class MemoryPartition:
    def __init__(
        self, size_in_mb: int, process: Optional[Process] = None, start: int = 0
    ):
        self.process = process
        self.start = start
        self.size_in_mb = size_in_mb

    @property
    def is_available(self) -> bool:
        return self.process is None


class Memory:
    memory_size_in_mb = 32 * gb_in_mb

    def __init__(self) -> None:
        self.partitions = [MemoryPartition(size_in_mb=self.memory_size_in_mb)]


class CPU:
    time_quantum = 3

    def __init__(self) -> None:
        self.id = next(cpu_id)
        self.quantum_count = 0
        self.process: Optional[Process] = None

    def set_process(self, process: Optional[Process]) -> None:
        self.quantum_count = 0
        self.process = process

    def unset_process(self) -> None:
        self.process = None

    def execute(self) -> None:
        if not self.process:
            raise Exception("no process to execute")
        if self.process.phase_1_duration > 0:
            self.process.phase_1_duration -= 1
        elif self.process.phase_2_duration > 0:
            self.process.phase_2_duration -= 1
        self.quantum_count += 1


class Viewer:
    def __init__(self, memory: Memory, queues: list[ProcessQueue]) -> None:
        self.memory = memory
        self.queues = queues
        self.process_versions_by_time: defaultdict[int, list[Process]] = defaultdict(
            list
        )
        self.disk_versions_by_time: defaultdict[int, list[Disk]] = defaultdict(list)
        self.cpu_versions_by_time: defaultdict[int, list[CPU]] = defaultdict(list)

    def save_process_version(self, process: Process, time: int) -> None:
        self.process_versions_by_time[time].append(deepcopy(process))

    def save_disks_version(self, disks: list[Disk], time: int) -> None:
        for disk in disks:
            self.disk_versions_by_time[time].append(deepcopy(disk))

    def save_cpu_version(self, cpu: CPU, time: int) -> None:
        self.cpu_versions_by_time[time].append(deepcopy(cpu))

    @property
    def pids(self) -> set[int]:
        processes = chain.from_iterable(list(self.process_versions_by_time.values()))
        return set([p.id for p in processes])

    @property
    def disk_ids(self) -> set[int]:
        disks = chain.from_iterable(list(self.disk_versions_by_time.values()))
        return set([d.id for d in disks])

    @property
    def cpu_ids(self) -> set[int]:
        cpus = chain.from_iterable(list(self.cpu_versions_by_time.values()))
        return set([c.id for c in cpus])

    @property
    def max_time(self) -> int:
        return max(list(self.process_versions_by_time.keys()))

    @property
    def simple_view_table(self) -> Table:
        table = Table(
            title="Simple Visualization", title_justify="center", min_width=30
        )
        table.add_column("T", justify="center", style="cyan", no_wrap=True)

        for pid in self.pids:
            table.add_column(f"P{pid}", justify="center", style="cyan", no_wrap=True)

        for time in range(0, self.max_time + 1):
            values = []
            for pid in self.pids:
                cpu_usages = [
                    cpu
                    for cpu in self.cpu_versions_by_time[time]
                    if cpu.process and cpu.process.id == pid
                ]
                disk_usages = [
                    d
                    for d in self.disk_versions_by_time[time]
                    if d.process and d.process.id == pid
                ]

                if cpu_usages:
                    values.append("X")
                elif disk_usages:
                    values.append("IO")
                else:
                    values.append(" ")
            table.add_row(str(time), *values)
        return table

    @property
    def process_states_table(self) -> Table:
        table = Table(
            title="Process States Timeline", title_justify="center", min_width=30
        )
        table.add_column("T", justify="center", style="cyan", no_wrap=True)

        for pid in self.pids:
            table.add_column(f"P{pid}", justify="center", style="cyan", no_wrap=True)

        for time in range(0, self.max_time + 1):
            values = []
            for pid in self.pids:
                process_versions = [
                    p for p in self.process_versions_by_time[time] if p.id == pid
                ]
                if process_versions:
                    state_changes = list(
                        dict.fromkeys([p.state.value for p in process_versions])
                    )
                    operation = " -> ".join([h for h in state_changes])
                else:
                    operation = ""
                values.append(operation)
            table.add_row(str(time), *values)
        return table

    @property
    def disks_table(self) -> Table:
        table = Table(title="Disks Timeline", title_justify="center", min_width=30)
        table.add_column("T", justify="center", style="cyan", no_wrap=True)

        for disk_id in self.disk_ids:
            table.add_column(
                f"Disk {disk_id}", justify="center", style="cyan", no_wrap=True
            )

        for time in range(0, self.max_time + 1):
            values = []
            for disk_id in self.disk_ids:
                disk_versions_with_processes = [
                    d
                    for d in self.disk_versions_by_time[time]
                    if d.id == disk_id and d.process is not None
                ]
                if disk_versions_with_processes:
                    assert disk_versions_with_processes[0].process
                    values.append(str(disk_versions_with_processes[0].process.id))
                else:
                    values.append(" ")
            table.add_row(str(time), *values)
        return table

    @property
    def cpus_table(self) -> Table:
        table = Table(title="CPUs Timeline", title_justify="center", min_width=30)
        table.add_column("T", justify="center", style="cyan", no_wrap=True)
        for cpu_id in self.cpu_ids:
            table.add_column(
                f"CPU {cpu_id}", justify="center", style="cyan", no_wrap=True
            )

        for time in range(0, self.max_time + 1):
            values = []
            for cpu_id in self.cpu_ids:
                cpu_versions_with_processes = [
                    cpu
                    for cpu in self.cpu_versions_by_time[time]
                    if cpu.id == cpu_id and cpu.process is not None
                ]
                if cpu_versions_with_processes:
                    assert cpu_versions_with_processes[0].process
                    values.append(str(cpu_versions_with_processes[0].process.id))
                else:
                    values.append(" ")
            table.add_row(str(time), *values)
        return table

    @property
    def memory_table(self) -> Table:
        table = Table(title="Memory", title_justify="center", min_width=30)
        table.add_column("Start", justify="center", style="cyan", no_wrap=True)
        table.add_column("End", justify="center", style="cyan", no_wrap=True)
        table.add_column("PID", justify="center", style="cyan", no_wrap=True)

        for partition in self.memory.partitions:
            pid = ""
            if partition.process:
                pid = str(partition.process.id)
            values = [
                str(partition.start),
                str(partition.start + partition.size_in_mb),
                pid,
            ]
            table.add_row(*values)
        return table

    @property
    def queues_table(self) -> Table:
        table = Table(title="Queues", title_justify="center", min_width=30)
        table.add_column("Queue", justify="center", style="cyan", no_wrap=True)
        table.add_column("Processes", justify="center", style="cyan", no_wrap=True)

        for index, queue in enumerate(self.queues):
            pids = [str(p.id) for p in queue.processes]
            value = ", ".join(pids)
            table.add_row(str(index), value)

        return table

    def print(self) -> None:
        input("Pressione qualquer tecla para continuar")
        os.system("clear")

        console = Console()
        panel_1 = Panel.fit(
            Columns(
                [
                    self.simple_view_table,
                    self.process_states_table,
                    self.disks_table,
                    self.cpus_table,
                ],
                padding=(1, 2),
            ),
            padding=(1, 2),
        )
        panel_2 = Panel.fit(
            Columns(
                [
                    self.memory_table,
                    self.queues_table,
                ],
                padding=(1, 2),
            ),
            padding=(1, 2),
        )
        console.print(panel_1, panel_2)


class Scheduler:
    number_of_queues = 4

    def __init__(self, cpus: list[CPU], memory: Memory, disks: list[Disk]):
        self.counter = 0
        self.disks: list[Disk] = disks
        self.cpus: list[CPU] = cpus
        self.memory: Memory = memory
        self.ready_queues: list[ProcessQueue] = [
            ProcessQueue() for _ in range(self.number_of_queues)
        ]
        self.blocked_processes: list[Process] = []
        self.suspended_processes: list[Process] = []

        self.viewer = Viewer(self.memory, self.ready_queues)

    @property
    def completed(self) -> bool:
        return (
            all([q.empty for q in self.ready_queues])
            and all([cpu.process is None for cpu in self.cpus])
            and not self.blocked_processes
            and not self.suspended_processes
        )

    @property
    def processes(self) -> list[Process]:
        ready_processes = list(
            chain.from_iterable([queue.processes for queue in self.ready_queues])
        )
        cpu_processes = [cpu.process for cpu in self.cpus if cpu.process is not None]
        return list(
            set(
                ready_processes
                + self.blocked_processes
                + self.suspended_processes
                + cpu_processes
            )
        )

    def try_to_allocate(self, process: Process) -> None:
        available_partitions = [p for p in self.memory.partitions if p.process is None]
        available_memory = sum([p.size_in_mb for p in available_partitions])
        for partition in available_partitions:
            if partition.size_in_mb > process.size_in_mb:
                new_partition = MemoryPartition(
                    size_in_mb=process.size_in_mb,
                    process=process,
                    start=partition.start,
                )
                partition.start += process.size_in_mb
                partition.size_in_mb -= process.size_in_mb
                self.memory.partitions.append(new_partition)
                self.memory.partitions = sorted(
                    self.memory.partitions, key=lambda p: p.start
                )
                process.allocated_in_memory = True
                return
            if partition.size_in_mb == process.size_in_mb:
                partition.process = process
                process.allocated_in_memory = True
                return

        blocked_processes_partitions = [
            p
            for p in self.memory.partitions
            if p.process is not None and p.process.state == ProcessState.BLOCKED
        ]
        for partition in blocked_processes_partitions:
            if partition.size_in_mb >= process.size_in_mb:
                self.suspend_process(assert_is_not_none(partition.process))
                self.deallocate(assert_is_not_none(partition.process))
                self.try_to_allocate(process)
                if process.allocated_in_memory:
                    self.suspended_processes = [
                        p for p in self.suspended_processes if p is not process
                    ]
                    return

        if not process.allocated_in_memory and available_memory >= process.size_in_mb:
            self.reallocate()
            self.try_to_allocate(process)
            return

        blocked_processes_memory = sum([p.size_in_mb for p in blocked_processes_partitions])
        try:
            if (available_memory + blocked_processes_memory) >= process.size_in_mb:
                i = 0
                while available_memory < process.size_in_mb:
                    partition = blocked_processes_partitions[i]
                    self.suspend_process(assert_is_not_none(partition.process))
                    self.deallocate(assert_is_not_none(partition.process))
                    available_partitions = [p for p in self.memory.partitions if p.process is None]
                    available_memory = sum([p.size_in_mb for p in available_partitions])
                    i += 1
                self.reallocate()
                self.try_to_allocate(process)
        except IndexError:
            return

    def reallocate(self) -> None:
        old_partitions = self.memory.partitions
        self.memory.partitions = [
            MemoryPartition(size_in_mb=self.memory.memory_size_in_mb)
        ]
        for partition in old_partitions:
            if partition.process:
                self.try_to_allocate(partition.process)

    def deallocate(self, process: Process) -> None:
        try:
            index, used_partition = [
                (index, p)
                for index, p in enumerate(self.memory.partitions)
                if p.process is process
            ][0]
        except IndexError:
            raise Exception("Process is not allocated")
        process.allocated_in_memory = False
        left_partition = None
        if index - 1 >= 0:
            left_partition = self.memory.partitions[index - 1]

        try:
            right_partition = self.memory.partitions[index + 1]
        except IndexError:
            right_partition = None

        if left_partition and left_partition.is_available:
            left_partition.size_in_mb += process.size_in_mb
            self.memory.partitions.remove(used_partition)
            if right_partition and right_partition.is_available:
                left_partition.size_in_mb += right_partition.size_in_mb
                self.memory.partitions.remove(right_partition)
            return

        used_partition.process = None
        if right_partition and right_partition.is_available:
            used_partition.size_in_mb += right_partition.size_in_mb
            self.memory.partitions.remove(right_partition)
            return

    def allocated_disks(self, process: Process) -> list[Disk]:
        return [d for d in self.disks if d.process is process]

    def save_process_version(self, process: Process) -> None:
        self.viewer.save_process_version(process, self.counter)

    def save_disks_version(self, disks: list[Disk]) -> None:
        self.viewer.save_disks_version(disks, self.counter)

    def save_cpu_version(self, cpu: CPU) -> None:
        self.viewer.save_cpu_version(cpu, self.counter)

    def add(self, process: Process) -> None:
        self.save_process_version(process)
        self.try_to_allocate(process)
        if process.allocated_in_memory:
            self.make_process_ready(process)
        else:
            self.suspend_process(process)

    def get_prioritary_process_from_queues(self) -> Optional[Process]:
        for index, queue in enumerate(self.ready_queues):
            try:
                eligible_process = queue.processes[0]
                if (
                    eligible_process
                    and eligible_process.last_executed_time < self.counter
                ):
                    process = queue.pop()
                    process.actual_queue = index
                    return process
            except IndexError:
                continue
        return None

    def block_process(self, process: Process) -> None:
        process.state = ProcessState.BLOCKED
        self.blocked_processes.append(process)
        self.save_process_version(process)

    def suspend_process(self, process: Process) -> None:
        if process.state == ProcessState.BLOCKED:
            process.state = ProcessState.SUSPENDED_BLOCKED
        elif process.state in (ProcessState.SUSPENDED_BLOCKED, ProcessState.NEW):
            process.state = ProcessState.SUSPENDED_READY
        if process not in self.suspended_processes:
            self.suspended_processes.append(process)
        self.save_process_version(process)

    def make_process_ready(self, process: Process) -> None:
        if process in self.blocked_processes:
            self.blocked_processes = [
                p for p in self.blocked_processes if p is not process
            ]
        if process.state == ProcessState.SUSPENDED_BLOCKED:
            process.state = ProcessState.SUSPENDED_READY
        else:
            process.state = ProcessState.READY
            self.ready_queues[0].push(process)

        self.save_process_version(process)

    def try_to_allocate_disks(self, process: Process) -> None:
        available_disks = [d for d in self.disks if d.process is None]
        if len(available_disks) >= process.disk_quantity:
            disks_to_allocate = available_disks[: process.disk_quantity]
            for disk in disks_to_allocate:
                disk.process = process

    def iterate(self) -> None:
        for process in self.processes:
            self.save_process_version(process)

        for process in self.suspended_processes:
            if process.state != ProcessState.SUSPENDED_BLOCKED:
                self.try_to_allocate(process)
                if process.allocated_in_memory:
                    self.make_process_ready(process)
                    self.suspended_processes = [
                        p for p in self.suspended_processes if p != process
                    ]

        processes_to_move_to_next_queue = defaultdict(list)
        for cpu in self.cpus:
            if cpu.quantum_count == cpu.time_quantum or not cpu.process:
                next_process = self.get_prioritary_process_from_queues()

                if next_process:
                    self.save_process_version(next_process)
                    next_process.state = ProcessState.RUNNING
                    cpu.set_process(next_process)

            if cpu.process:
                cpu.execute()
                self.save_cpu_version(cpu)
                cpu.process.state = ProcessState.RUNNING
                cpu.process.last_executed_time = self.counter
                self.save_process_version(cpu.process)
                if cpu.process.phase_1_duration == 0 and cpu.process.io_duration > 0:
                    self.block_process(cpu.process)
                    cpu.unset_process()

                elif cpu.process.phase_2_duration == 0:
                    cpu.process.state = ProcessState.EXIT
                    self.deallocate(cpu.process)
                    self.save_process_version(cpu.process)
                    cpu.unset_process()
                    continue

                elif cpu.quantum_count == cpu.time_quantum:
                    processes_to_move_to_next_queue[cpu.process.actual_queue].append(
                        cpu.process
                    )
                    cpu.unset_process()

        for actual_queue, processes in processes_to_move_to_next_queue.items():
            processes = sorted(processes, key=lambda p: p.remaining_time)
            for process in processes:
                try:
                    self.ready_queues[process.actual_queue + 1].push(process)
                    process.actual_queue += 1
                except IndexError:
                    self.ready_queues[process.actual_queue].push(process)
                process.state = ProcessState.READY
                self.save_process_version(process)

        self.save_disks_version(self.disks)
        disks_to_deallocate = []
        for process in self.blocked_processes:
            if process.last_executed_time == self.counter:
                continue
            self.save_process_version(process)
            if len(self.allocated_disks(process)) != process.disk_quantity:
                self.try_to_allocate_disks(process)
            if len(self.allocated_disks(process)) == process.disk_quantity:
                process.wait_for_io()
                self.save_process_version(process)
                if process.io_duration == 0:
                    self.make_process_ready(process)
                    for disk in self.allocated_disks(process):
                        disks_to_deallocate.append(disk)

        for disk in disks_to_deallocate:
            disk.deallocate()
        self.save_disks_version(self.disks)
        self.viewer.print()
        self.counter += 1


def read_processes_from_file() -> list[Process]:
    processes = []
    file_name = input("Digite o nome do arquivo com os processos: ")
    with open(file_name, "r") as f:
        for line in f:
            (
                arrival,
                phase_1_duration,
                io_duration,
                phase_2_duration,
                size_in_mb,
                disk_quantity,
            ) = [int(i) for i in line.split(", ")]
            process = Process(
                id=next(process_id),
                arrival=arrival,
                phase_1_duration=phase_1_duration,
                io_duration=io_duration,
                phase_2_duration=phase_2_duration,
                size_in_mb=size_in_mb,
                disk_quantity=disk_quantity,
            )
            processes.append(process)
    return sorted(processes, key=lambda p: p.arrival)


def main() -> None:
    cpus = [CPU() for _ in range(4)]
    memory = Memory()
    disks = [Disk() for _ in range(4)]
    processes_to_schedule = read_processes_from_file()

    scheduler = Scheduler(cpus, memory, disks)

    while processes_to_schedule or not scheduler.completed:
        for process in processes_to_schedule:
            if process.arrival > scheduler.counter:
                break
            elif process.arrival == scheduler.counter:
                processes_to_schedule = [
                    p for p in processes_to_schedule if p != process
                ]
                scheduler.add(process)
        scheduler.iterate()


if __name__ == "__main__":
    main()

# 1. Functional Requirements (FR)
- FR-1: Alert Generation: The system must be able to receive medical data, process it, and generate an alert for a user based on the processing outcome.
- FR-2: The system must apply a valid algorithm to calculate a risk score for chronic kidney disease

# 2. Non-Functional Requirements (NFR)
## 2.1. Adaptability / Flexibility
- NFR-2.1.1: Input Adaptability: The system must be easily adaptable to different types of input data formats.
- NFR-2.1.2: Processing Model Swap: The system must allow the data processing model (e.g., the algorithm or logic) to be changed or updated without requiring the system to be stopped or taken offline.
- NFR-2.1.3: Output Adaptability: The system must be capable of easily adapting its output format or structure.
- NFR-2.1.4: Operating System Compatibility: The system must be capable of executing on different operating systems (e.g., Windows, Linux, macOS).
- NFR-2.1.5: Execution Modes: The system must be adaptable to different execution modes (e.g., scan/polling, batches, event-driven, flag-based triggers).

## 2.2. Performance / Time Constraints
- NFR-2.2.1: Real-Time Processing: The system must be capable of processing data in real time.
- NFR-2.2.2: Real-Time Response: The system must be capable of responding (generating the alert) in real time.

## 2.3. Availability / Reliability
- NFR-2.3.1: Availability: The system must be accessible and operational 99.9% of the time (equivalent to an acceptable amount of downtime per year).

## 2.4. Scalability / Capacity
- NFR-2.4.1: Scalability: The system must be scalable to meet the evolving data and user needs of the health system.

## 2.5. Deployment / Environmenta
- NFR-2.5.1: Deployment Flexibility: The system must be capable of being deployed both on-premise and on the cloud.

## 2.6. Security / Data Handling
- NFR-2.6.1: Data Persistence Restriction: The system must not store or persist any of the processed data (i.e., it must be stateless).


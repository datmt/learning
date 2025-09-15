

# **Code-First Networking Mastery: An Annotated Curriculum and Implementation Guide**

This guide serves as the definitive companion to the "Code-First Networking Mastery" curriculum. Its philosophy is rooted in the belief that true mastery of networking comes not from memorization, but from implementation. By building the core components of network infrastructure—from the raw bytes of a single packet to the complex logic of a service mesh—you will gain an intuitive, first-principles understanding of how modern networks operate. This document will provide the detailed code, conceptual background, and expert insights necessary to transform you from a programmer who *uses* the network to an architect who *understands* it.

## **Part I: The Building Blocks of Network Communication**

### **Module 1: Raw Socket Fundamentals**

**Core Objective:** To establish a foundational understanding of network communication at the lowest practical level in Python, using the standard socket library. This module forces the learner to confront the raw, unstructured nature of network data streams, building an appreciation for the abstractions that higher-level protocols and libraries provide.

#### **1.1 Basic Socket Creation (UDP)**

The journey begins with the User Datagram Protocol (UDP), a fundamental transport layer protocol defined in **RFC 768**.1 UDP is a connectionless protocol, meaning it does not establish a persistent connection between client and server before sending data.2 This "fire-and-forget" nature results in low overhead and high speed, making it suitable for applications like video streaming, online gaming, and DNS queries where occasional packet loss is acceptable.3 However, this performance comes at the cost of reliability; UDP provides no guarantees that packets will arrive in order, or even that they will arrive at all.5

##### **UDP Server Implementation**

A UDP server's primary role is to bind to a specific network address and port, and then enter a loop to listen for incoming datagrams. The implementation in Python relies on the socket module.4

The core steps are:

1. **Socket Creation:** A socket object is created using socket.socket(). The arguments socket.AF_INET and socket.SOCK_DGRAM specify the use of the IPv4 address family and the UDP protocol, respectively.7  
2. **Binding:** The bind() method associates the newly created socket with a specific IP address and port number on the host machine. This is the address the server will "listen" on.7  
3. **Receiving Data:** The server enters an infinite loop, blocking on the recvfrom() call. This method waits for a datagram to arrive. Upon arrival, it returns a tuple containing the payload (as bytes) and the address (IP, port) of the client that sent it.3

*udp_server.py*

```python

# udp_server.py  
import socket

# Define the server's IP address and port  
UDP_IP = "127.0.0.1"  # Loopback address for localhost  
UDP_PORT = 8000

# 1. Create a UDP socket  
#    AF_INET specifies the use of IPv4 addresses.  
#    SOCK_DGRAM specifies that this is a UDP socket.  
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. Bind the socket to the server's address and port  
sock.bind((UDP_IP, UDP_PORT))  
print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")

# 3. Enter a loop to continuously listen for messages  
while True:  
    # Wait to receive data from a client. The buffer size is 1024 bytes.  
    # recvfrom() returns the data payload and the address of the sender.  
    data, addr = sock.recvfrom(1024)   
      
    # Print the received message and the client's address  
    print(f"Received message: '{data.decode()}' from {addr}")
```
##### **UDP Client Implementation**

A UDP client is simpler than a server. It does not need to bind to a specific port (the operating system will assign an ephemeral port automatically). Its only job is to create a socket and send a datagram to the server's known address and port using the sendto() method.3

```python
# udp_client.py  
import socket

# Define the server's address and the message to send  
UDP_IP = "127.0.0.1"  
UDP_PORT = 8000  
MESSAGE = b"Hello, World!"

# 1. Create a UDP socket  
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending message: '{MESSAGE.decode()}' to {UDP_IP}:{UDP_PORT}")

# 2. Send the message to the server's address and port  
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

# 3. Close the socket  
sock.close()
```
##### **Testing and Verification**

To test the communication, first run udp_server.py in one terminal. It will print the listening address and wait. Then, run udp_client.py in a second terminal. The client will send its message and exit. The server terminal will then print the message it received and the client's address, confirming successful communication.

#### **1.2 TCP Connection Basics**

In contrast to UDP, the Transmission Control Protocol (TCP), defined in **RFC 793**, is a connection-oriented protocol.10 It provides reliable, ordered, and error-checked delivery of a stream of bytes between applications.2 Before any application data is exchanged, TCP establishes a connection via a process known as the three-way handshake (SYN, SYN-ACK, ACK). This handshake ensures both client and server are ready and able to communicate, forming the basis for reliable data transfer used by protocols like HTTP, FTP, and SSH.9

##### **Implementing a TCP Client for HTTP**

This task involves creating a TCP client that connects to a web server and sends a raw HTTP request. This exercise demonstrates the stream-based nature of TCP and the text-based structure of HTTP.

The process involves:

1. **Socket Creation:** A TCP socket is created using socket.SOCK_STREAM.9  
2. **Connection:** The connect() method is called to initiate the three-way handshake with the target server on a specific port (port 80 for HTTP).12  
3. **Request Crafting:** A valid HTTP GET request is constructed as a byte string. The HTTP protocol standard requires lines to be terminated by a carriage return and line feed (\r\n), and the entire header section must be terminated by a blank line (\r\n\r\n).13  
4. **Sending Data:** The sendall() method is used to ensure the entire request is transmitted.11  
5. **Receiving Data:** Because TCP provides an unstructured stream of bytes, a single recv() call is not guaranteed to receive the entire server response. Data must be received in a loop until the server closes the connection, indicated by recv() returning an empty byte string.9 This implementation detail is a direct consequence of TCP's stream abstraction, which contrasts sharply with UDP's datagram model. While a single  
   recvfrom() call in the UDP server retrieves one complete message, the TCP client must buffer incoming data until the transmission is complete. This fundamental difference has significant implications for application design, influencing everything from message parsing to session management.

*tcp_http_client.py*

```python

# tcp_http_client.py  
import socket

TARGET_HOST = "httpbin.org"  
TARGET_PORT = 80

# 1. Create a TCP socket (SOCK_STREAM for TCP)  
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:  
    # 2. Connect to the server, initiating the TCP 3-way handshake  
    client.connect((TARGET_HOST, TARGET_PORT))  
    print(f"Connected to {TARGET_HOST}:{TARGET_PORT}")

    # 3. Craft the raw HTTP GET request as a byte string  
    # Note the \r\n line endings and the final \r\n\r\n to end the headers  
    request = b"GET /get HTTP/1.1\r\nHost: httpbin.org\r\nConnection: close\r\n\r\n"  
      
    # 4. Send the entire request  
    client.sendall(request)  
    print("Sent HTTP request.")

    # 5. Receive the response in a loop until the connection is closed  
    response = b""  
    while True:  
        # Read data in chunks of 4096 bytes  
        chunk = client.recv(4096)  
        if not chunk:  
            # If recv returns an empty string, the server has closed the connection  
            break  
        response += chunk  
      
    print("\n--- HTTP Response ---")  
    # Decode for printing, ignoring potential decoding errors for non-text parts  
    print(response.decode(errors='ignore'))  
    print("--------------------")

finally:  
    # Ensure the socket is closed  
    client.close()  
    print("Connection closed.")
```
#### **1.3 Packet Observation**

Theoretical knowledge of protocol differences becomes concrete when observed on the wire. Using a network protocol analyzer like Wireshark provides invaluable visual confirmation of the concepts implemented in code.9

##### **Guidance for Analysis**

1. **Start Capture:** Launch Wireshark and begin capturing on the appropriate network interface. For the UDP and TCP clients communicating with localhost (127.0.0.1), this will be the **loopback interface** (often named lo, lo0, or "Loopback Traffic"). For the HTTP client connecting to httpbin.org, this will be your machine's primary network interface (e.g., eth0 or en0).  
2. **Apply Filters:** To isolate the relevant traffic, use Wireshark's display filters.  
   * For the UDP test: udp.port == 8000.15  
   * For the TCP test: tcp.port == 80 and ip.addr == <ip_of_httpbin.org>.15  
3. **Execute Scripts:** Run the Python server and client scripts while the capture is active.  
4. **Analyze the Flows:**  
   * **UDP Flow:** The capture will show a simple two-packet exchange: one packet from the client to the server, and no response from the example server. This visually demonstrates the connectionless nature of UDP.  
   * **TCP Flow:** The capture will be much more detailed. Right-click on one of the TCP packets and select "Follow > TCP Stream" to see the full conversation. The main window will show the distinct phases:  
     * **Handshake:** Three packets with flags , , and [ACK]. This is the connection establishment.  
     * **Data Transfer:** A packet with the flags, containing the HTTP \`GET\` request payload. The server will respond with its own series of \`[ACK]\` and packets containing the HTTP response.  
     * **Teardown:** A series of packets with [FIN, ACK] and [ACK] flags, showing the graceful closing of the connection from both sides.

This comparative analysis makes the abstract differences between UDP and TCP tangible, showing how TCP's reliability features manifest as additional packets on the network.

#### **1.4 Raw Socket Introduction**

The final task of this module descends another layer in the network stack. Standard sockets (SOCK_STREAM, SOCK_DGRAM) operate at the transport layer, delegating the creation of TCP and UDP headers to the operating system's kernel. Raw sockets (SOCK_RAW) provide a powerful interface to bypass this behavior, allowing a program to construct its own transport-layer (or even network-layer) headers.16 This capability is the foundation of many advanced networking tools, including port scanners and packet crafters, but requires administrative privileges to use.

The progression from standard sockets to raw sockets is a critical step in deconstructing the network stack. It shifts the programmer's perspective from being a consumer of the transport layer's services to being a constructor of its protocols. This hands-on creation of headers using binary packing demystifies what is otherwise a black box, providing a foundational understanding that is indispensable for advanced network programming and security analysis.

##### **Manual Header Creation with struct.pack()**

To build a protocol header in Python, one must convert high-level data like integers into a specific sequence of bytes. The struct module is the standard tool for this task, packing Python values into C-style structs, which are contiguous blocks of binary data.17

The UDP header, as per RFC 768, consists of four 16-bit (2-byte) fields. The following table maps these fields to their struct format string representation.

| Field Name | Size (bytes) | struct Format | Description | RFC 768 Reference |
| :---- | :---- | :---- | :---- | :---- |
| Source Port | 2 | H | Port of the sending process. | Section "Fields" |
| Destination Port | 2 | H | Port of the receiving process. | Section "Fields" |
| Length | 2 | H | Length of UDP header + data in bytes. | Section "Fields" |
| Checksum | 2 | H | Optional error-checking field. | Section "Fields" |

The format string !HHHH is used with struct.pack():

* !: Specifies network byte order (big-endian), which is the standard for network protocols.  
* H: Represents an unsigned short integer (2 bytes), matching the size of each UDP header field.

##### **Annotated Code**

The following script creates a raw socket, manually builds a UDP header, attaches a payload, and sends the complete packet. The operating system's kernel will still be responsible for prepending the necessary IP header before sending the packet on the wire.

*raw_udp_sender.py*

```python

# raw_udp_sender.py  
# NOTE: This script must be run with sudo/administrator privileges.  
import socket  
import struct

# --- Manually build the UDP header ---  
# Arbitrary source port  
source_port = 12345  
# Destination port for our UDP server  
dest_port = 8000  
# Payload data  
payload = b"Manually Crafted Packet"  
# Length is the size of the UDP header (8 bytes) plus the payload size  
length = 8 + len(payload)  
# Checksum can be set to 0; the kernel will calculate it if IPPROTO_UDP is used.  
checksum = 0

# 1. Pack the header fields into a byte string.  
#   ! = Network Byte Order (Big-Endian)  
#    H = Unsigned Short (2 bytes)  
#    The format '!HHHH' packs four unsigned shorts.  
udp_header = struct.pack('!HHHH', source_port, dest_port, length, checksum)

# 2. Create a raw socket.  
#    AF_INET for IPv4.  
#    SOCK_RAW indicates a raw socket.  
#    IPPROTO_UDP tells the kernel that the data we provide (header + payload)  
#    should be encapsulated within an IP packet with the protocol number for UDP (17).  
try:  
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_UDP)  
except PermissionError:  
    print("Permission denied. Please run this script with sudo or as an administrator.")  
    exit()

# The kernel will automatically create the IP header.  
# If we wanted to build the IP header ourselves, we would use IPPROTO_RAW  
# and set the IP_HDRINCL socket option:  
# s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1\)

# 3. Send the packet.  
#    The payload is our manually built UDP header plus the application data.  
#    The destination address is provided to sendto(), which the kernel uses  
#    to build the destination IP field in the IP header.  
destination_address = ('127.0.0.1', dest_port)  
s.sendto(udp_header + payload, destination_address)

print("Sent a manually crafted UDP packet.")  
s.close()
```
To verify this, run the udp_server.py from Task 1.1 in one terminal. In another terminal, run raw_udp_sender.py with sudo. The server will print the received message, confirming that the manually constructed packet was successfully received and processed. A Wireshark capture on the loopback interface will show a valid UDP packet with the source port set to 12345.

### **Module 2: Packet Crafting with Scapy**

**Core Objective:** To transition from manual, cumbersome packet creation with struct to the powerful, expressive, and flexible Scapy library. This module introduces the concept of protocol layers as programmable objects, enabling the rapid creation and dissection of complex packets.

#### **2.1 Scapy Basics**

Scapy is a powerful Python-based packet manipulation program and library that enables the user to forge, decode, send, and sniff network packets.18 It can replace a wide array of traditional networking tools like

ping, nmap, and tcpdump by providing a single, programmatic interface.19 Its core strength lies in its intuitive model of treating protocol layers as stackable objects. The

/ operator is used to combine layers, and Scapy automatically handles details like setting protocol types in lower layers and calculating checksums.21

##### **Implementation: ICMP Ping**

This task replicates the functionality of the ping utility by crafting and sending an ICMP Echo Request packet.

1. **Start Scapy:** Launch the interactive shell with administrative privileges: sudo scapy.  
2. **Craft Packet:** Create an ICMP packet destined for Google's public DNS server (8.8.8.8). The IP() layer defines the network-level addressing, and the ICMP() layer defines the protocol payload.18  
3. **Inspect Packet:** Use ls() to view the fields of the crafted packet. Scapy automatically fills in reasonable defaults, such as the source IP address based on the egress interface's configuration.24  
4. **Send and Receive:** Use the sr1() function to send the packet at Layer 3 and wait for a single response. The timeout parameter prevents the function from waiting indefinitely.22  
5. **Analyze Response:** If a response is received, it is a Scapy packet object that can be inspected. The summary() method provides a one-line description, and individual fields can be accessed using dictionary-like syntax (e.g., response[IP].src). An ICMP Echo Reply has a type code of 0.

*scapy_ping.py*

```python

# scapy_ping.py  
from scapy.all import IP, ICMP, sr1, ls

# 1. Craft the packet by stacking layers.  
#    Scapy automatically fills in default values like the source IP address.  
packet = IP(dst="8.8.8.8") / ICMP()

print("--- Packet to be Sent ---")  
# The show() method provides a detailed view of the packet's layers and fields.  
packet.show()  
print("-------------------------")

print("\nSending ICMP Echo Request...")  
# 2. Send the packet at Layer 3 and wait for a single reply (sr1).  
#    verbose=0 suppresses the default send/receive messages.  
#    timeout=2 sets a 2-second timeout.  
response = sr1(packet, timeout=2, verbose=0)

# 3. Analyze the response.  
if response:  
    print("\n--- Response Received ---")  
    response.show()  
    print("-----------------------")  
      
    print("\nResponse Summary:")  
    print(response.summary())  
      
    print(f"\nSource IP of Response: {response[IP].src}")  
    # An ICMP Echo Reply has a type code of 0.  
    if response.haslayer(ICMP) and response[ICMP].type == 0:  
        print("Received a valid ICMP Echo Reply.")  
else:  
    print("\nNo response received within the timeout period.")
```
#### **2.2 Layer Composition**

The true power of Scapy's layering model is the ability to precisely control every field in a packet. This task demonstrates this control by manually performing a TCP three-way handshake, a process normally handled entirely by the operating system. This requires crafting three distinct packets (SYN, SYN-ACK, ACK) and managing the sequence and acknowledgment numbers correctly.

##### **Implementation: Manual TCP Handshake**

1. **Send SYN:** A TCP packet with the S (SYN) flag is created and sent to the target's port 80. The sr1() function is used to capture the server's response.22  
2. **Receive and Analyze SYN-ACK:** The response is checked to ensure it is a TCP packet with the SA (SYN-ACK) flags set. The server's sequence number (seq) and acknowledgment number (ack) are extracted from this packet. These values are crucial for constructing the final ACK packet.  
3. **Send ACK:** The final packet of the handshake is created with the A (ACK) flag. Its sequence number is set to the server's acknowledgment number, and its acknowledgment number is set to the server's sequence number plus one. This completes the handshake. The send() function is used here because no reply is expected for an ACK packet.

*manual_handshake.py*

```python

# manual_handshake.py  
from scapy.all import IP, TCP, sr1, send

# Target information  
target_ip = "scanme.nmap.org" # A host specifically for testing scanners  
target_port = 80

print(f"Attempting manual TCP handshake with {target_ip}:{target_port}")

# 1. Send SYN packet  
print("\nStep 1: Sending SYN packet...")  
# Craft a TCP packet with the SYN flag set ('S')  
# Use RandShort() for a random source port to avoid conflicts  
ip_layer = IP(dst=target_ip)  
tcp_syn = TCP(sport=RandShort(), dport=target_port, flags="S", seq=1000)  
packet_syn = ip_layer / tcp_syn

# Send the packet and wait for the first response  
syn_ack_response = sr1(packet_syn, timeout=2, verbose=0)

if not syn_ack_response:  
    print("No SYN-ACK response received. Target may be down or port is closed/filtered.")  
    exit()

# 2. Receive and Analyze SYN-ACK  
print("Step 2: Received SYN-ACK response.")  
syn_ack_response.show()

# Check for the presence of TCP layer and correct flags (0x12 means SYN+ACK)  
if not syn_ack_response.haslayer(TCP) or syn_ack_response.getlayer(TCP).flags!= 0x12:  
    print("Did not receive a valid SYN-ACK packet.")  
    exit()

# 3. Send ACK packet  
print("\nStep 3: Sending ACK packet...")  
# Extract sequence and acknowledgment numbers from the server's response  
server_seq = syn_ack_response.getlayer(TCP).seq  
server_ack = syn_ack_response.getlayer(TCP).ack

# Our ACK packet must acknowledge the server's sequence number  
# Our ACK number should be server_seq + 1  
# Our SEQ number should be the server's ACK number  
tcp_ack = TCP(sport=tcp_syn.sport, dport=target_port, flags="A",   
              seq=server_ack, ack=server_seq + 1)  
packet_ack = ip_layer / tcp_ack

# Send the final ACK packet. No response is expected.  
send(packet_ack, verbose=0)

print("Manual handshake completed successfully.")
```
Running this script while capturing in Wireshark will show the three distinct packets of the handshake, all originating from the script.

#### **2.3 Packet Analysis Script**

This task shifts from active packet sending to passive network listening. Scapy's sniff() function provides a powerful and simple way to capture network traffic, akin to tcpdump.23 The function can process packets in real-time using a callback function, which is specified with the

prn argument. This approach is highly memory-efficient for long-running captures, as setting store=False tells Scapy not to keep the packets in memory.26

This distinction between active (sr()) and passive (sniff()) functions introduces the two fundamental modes of network interaction. Active probing is the basis for reconnaissance tools that query the network to determine its state, as explored in Module 3. Passive listening is the foundation for monitoring and analysis tools that observe network behavior, which is the focus of Module 4. Understanding both paradigms is essential for a network architect.

##### **Implementation: Real-time Traffic Categorization**

The script sniffs network traffic for 60 seconds and provides a live-updating count of common protocols.

1. **Callback Function:** A function process_packet(packet) is defined. This function is executed by sniff() for every captured packet.  
2. **Protocol Identification:** Inside the callback, packet.haslayer() is used to check for the presence of different protocol layers, such as HTTPRequest, DNS, ICMP, etc.26  
3. **Counting:** A global dictionary, protocol_counts, is used to store the number of packets seen for each protocol.  
4. **Real-time Display:** After processing each packet, the script prints the updated dictionary to the console. The end='\r' argument in the print function moves the cursor to the beginning of the line without advancing to the next, creating a simple real-time update effect.  
5. **Sniffing Call:** The sniff() function is called with the prn argument pointing to the callback function. A filter="ip" is used to ignore non-IP traffic like ARP.

*packet_counter.py*

```python

# packet_counter.py  
from scapy.all import sniff, IP, TCP, UDP, ICMP, DNS  
from scapy.layers.http import HTTPRequest # Requires scapy_http layer  
import time  
import sys

# Dictionary to store protocol counts  
protocol_counts = {"HTTP": 0, "DNS": 0, "ICMP": 0, "TCP": 0, "UDP": 0, "Other IP": 0}

def process_packet(packet):  
    """Callback function to process each sniffed packet."""  
    # Check for highest-level protocols first  
    if packet.haslayer(HTTPRequest):  
        protocol_counts += 1  
    elif packet.haslayer(DNS):  
        protocol_counts += 1  
    # Then check for transport-level protocols  
    elif packet.haslayer(ICMP):  
        protocol_counts["ICMP"] += 1  
    elif packet.haslayer(TCP):  
        # This will count TCP packets that are not HTTP or DNS-over-TCP  
        protocol_counts += 1  
    elif packet.haslayer(UDP):  
        # This will count UDP packets that are not DNS  
        protocol_counts += 1  
    elif packet.haslayer(IP):  
        protocol_counts["Other IP"] += 1  
      
    # Format the output string for a clean, real-time display  
    status_line = " | ".join([f"{proto}: {count}" for proto, count in protocol_counts.items()])  
    sys.stdout.write("\r" + status_line)  
    sys.stdout.flush()

print("Starting packet sniffer for 60 seconds... (Browse the web to generate traffic)")  
# store=False prevents Scapy from keeping all packets in memory  
sniff(filter="ip", prn=process_packet, store=False, timeout=60)

print("\n\nSniffing complete. Final counts:")  
print(protocol_counts)
```

#### **2.4 Custom Protocol**

A key feature of Scapy is its extensibility. It allows developers to define their own protocol layers, which can then be used just like built-in protocols such as IP and TCP. This is accomplished by subclassing Scapy's Packet class and defining the protocol's structure in a special list called fields_desc.28

This exercise demonstrates that network protocols are not immutable; they are simply well-defined data structures. Scapy's object-oriented approach to layering directly mirrors the encapsulation model of network stacks.20 By defining a custom protocol, one gains a first-hand understanding of how these data structures are defined and interpreted.

##### **Implementation: Sender and Receiver for a Custom Protocol**

1. **Protocol Definition:** A new class, CustomProtocol, is created that inherits from Packet. The fields_desc attribute is a list of Field objects (ByteField, ShortField, etc.) that define the header's structure: a 1-byte version, a 1-byte type, and a 2-byte length.  
2. **Layer Binding:** The bind_layers() function is used to teach Scapy how to automatically dissect the new protocol. By calling bind_layers(UDP, CustomProtocol, dport=12345), we instruct Scapy that any UDP packet destined for port 12345 should have its payload dissected as a CustomProtocol layer.28  
3. **Sender Script:** The sender constructs a packet by stacking layers: IP()/UDP()/CustomProtocol()/Payload. It sets the custom header fields and sends the packet.  
4. **Receiver Script:** The receiver uses sniff() with a filter for UDP port 12345. The callback function checks if the received packet haslayer(CustomProtocol). Because of the bind_layers call, Scapy automatically dissects the payload, making the custom fields directly accessible (e.g., packet[CustomProtocol].version).

*custom_protocol_demo.py*

```python

# custom_protocol_demo.py  
from scapy.all import *  
import threading  
import time

# --- 1. Define the Custom Protocol Layer ---  
class CustomProtocol(Packet):  
    name = "CustomProtocol"  
    # Define the header fields: version(1 byte), type(1 byte), length(2 bytes)  
    fields_desc =

# --- 2. Bind the Layer ---  
# We tell Scapy that any UDP packet on destination port 12345  
# should have its payload interpreted as our CustomProtocol.  
bind_layers(UDP, CustomProtocol, dport=12345)

# --- 3. Receiver (Sniffer) Logic ---  
def receiver():  
    print(" Starting to sniff on UDP port 12345...")  
    # The prn function will be called for each packet that matches the filter  
    sniff(filter="udp port 12345", prn=process_custom_packet, store=False, timeout=10)  
    print(" Sniffing finished.")

def process_custom_packet(packet):  
    if packet.haslayer(CustomProtocol):  
        print("\n Custom Packet Detected!")  
          
        # Access the custom layer  
        custom_layer = packet[CustomProtocol]  
          
        print(f"  Source IP: {packet[IP].src}")  
        print(f"  Source Port: {packet.sport}")  
        print(f"  Header - Version: {custom_layer.version}")  
        print(f"  Header - Type: {custom_layer.type}")  
        print(f"  Header - Length: {custom_layer.length}")  
          
        # Check for a raw payload after our custom header  
        if packet.haslayer(Raw):  
            payload = packet.load.decode()  
            print(f"  Payload Data: '{payload}'")  
          
        # Verify length field matches payload  
        if custom_layer.length == len(packet.load):  
            print("  Length field is correct.")  
        else:  
            print("  Length field mismatch!")

# --- 4. Sender Logic ---  
def sender():  
    print(" Waiting 2 seconds before sending packet...")  
    time.sleep(2)  
      
    # Craft the packet  
    payload_data = b"This is my custom data"  
    packet = (  
        IP(dst="127.0.0.1") /  
        UDP(sport=RandShort(), dport=12345) /  
        CustomProtocol(version=2, type=1) /  
        Raw(load=payload_data)  
    )

    # Scapy automatically calculates the 'length' field in CustomProtocol  
    # because it knows the length of the Raw payload that follows it.  
      
    print(" Sending the following packet:")  
    packet.show()  
      
    send(packet, verbose=0)  
    print(" Packet sent.")

# --- Main Execution ---  
if __name__ == "__main__":  
    # Run the receiver in a separate thread so it can listen  
    # while the main thread prepares and sends the packet.  
    receiver_thread = threading.Thread(target=receiver)  
    receiver_thread.start()  
      
    # Run the sender in the main thread  
    sender()  
      
    # Wait for the receiver thread to finish its timeout  
    receiver_thread.join()  
    print("\n[Main] Demo finished.")
```
#### **Works cited**

1. RFC 768: User Datagram Protocol, accessed September 7, 2025, [https://www.rfc-editor.org/rfc/rfc768.html](https://www.rfc-editor.org/rfc/rfc768.html)  
2. Python Socket Programming: A 101 Guide of the Basics - Turing, accessed September 7, 2025, [https://www.turing.com/kb/socket-programming-in-python](https://www.turing.com/kb/socket-programming-in-python)  
3. How to Capture udp Packets in Python - GeeksforGeeks, accessed September 7, 2025, [https://www.geeksforgeeks.org/python/how-to-capture-udp-packets-in-python/](https://www.geeksforgeeks.org/python/how-to-capture-udp-packets-in-python/)  
4. Python Socket Programming: Server and Client Example Guide - DigitalOcean, accessed September 7, 2025, [https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client](https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client)  
5. UDP - Client and Server example programs in Python - Pythontic.com, accessed September 7, 2025, [https://pythontic.com/modules/socket/udp-client-server-example](https://pythontic.com/modules/socket/udp-client-server-example)  
6. Python - Socket Programming - Tutorials Point, accessed September 7, 2025, [https://www.tutorialspoint.com/python/python_socket_programming.htm](https://www.tutorialspoint.com/python/python_socket_programming.htm)  
7. A Complete Guide to Socket Programming in Python - DataCamp, accessed September 7, 2025, [https://www.datacamp.com/tutorial/a-complete-guide-to-socket-programming-in-python](https://www.datacamp.com/tutorial/a-complete-guide-to-socket-programming-in-python)  
8. UDP Communication - Python Wiki, accessed September 7, 2025, [https://wiki.python.org/moin/UdpCommunication](https://wiki.python.org/moin/UdpCommunication)  
9. Socket Programming in Python (Guide) – Real Python, accessed September 7, 2025, [https://realpython.com/python-sockets/](https://realpython.com/python-sockets/)  
10. RFC 793 - Transmission Control Protocol (TCP) - IETF, accessed September 7, 2025, [https://www.ietf.org/rfc/rfc793.txt](https://www.ietf.org/rfc/rfc793.txt)  
11. TCP/IP Client and Server - Python Module of the Week - PyMOTW 3, accessed September 7, 2025, [https://pymotw.com/2/socket/tcp.html](https://pymotw.com/2/socket/tcp.html)  
12. Socket Programming HOWTO — Python 3.13.7 documentation, accessed September 7, 2025, [https://docs.python.org/3/howto/sockets.html](https://docs.python.org/3/howto/sockets.html)  
13. Making HTTP requests with sockets in Python - Internal Pointers, accessed September 7, 2025, [https://www.internalpointers.com/post/making-http-requests-sockets-python.html](https://www.internalpointers.com/post/making-http-requests-sockets-python.html)  
14. Python TCP-Server - Medium, accessed September 7, 2025, [https://medium.com/@mando_elnino/python-tcp-server-b945c68a983c](https://medium.com/@mando_elnino/python-tcp-server-b945c68a983c)  
15. Using Wireshark for Packet analysis in Python - Studytonight, accessed September 7, 2025, [https://www.studytonight.com/network-programming-in-python/using-wireshark](https://www.studytonight.com/network-programming-in-python/using-wireshark)  
16. Raw Socket Programming in Python on Linux - Code Examples - BinaryTides, accessed September 7, 2025, [https://www.binarytides.com/raw-socket-programming-in-python-linux/](https://www.binarytides.com/raw-socket-programming-in-python-linux/)  
17. struct — Interpret bytes as packed binary data — Python 3.13.7 documentation, accessed September 7, 2025, [https://docs.python.org/3/library/struct.html](https://docs.python.org/3/library/struct.html)  
18. Scapy, accessed September 7, 2025, [https://scapy.net/](https://scapy.net/)  
19. Introduction — Scapy 2.6.1 documentation, accessed September 7, 2025, [https://scapy.readthedocs.io/en/latest/introduction.html](https://scapy.readthedocs.io/en/latest/introduction.html)  
20. Scapy Tutorial: WiFi Security, accessed September 7, 2025, [http://www.cs.toronto.edu/~arnold/427/18s/427_18S/indepth/scapy_wifi/scapy_tut.html](http://www.cs.toronto.edu/~arnold/427/18s/427_18S/indepth/scapy_wifi/scapy_tut.html)  
21. Creating packets - The Art of Packet Crafting with Scapy!, accessed September 7, 2025, [https://0xbharath.github.io/art-of-packet-crafting-with-scapy/scapy/creating_packets/index.html](https://0xbharath.github.io/art-of-packet-crafting-with-scapy/scapy/creating_packets/index.html)  
22. Usage — Scapy 2.6.1 documentation, accessed September 7, 2025, [https://scapy.readthedocs.io/en/latest/usage.html](https://scapy.readthedocs.io/en/latest/usage.html)  
23. Exploring Network Fundamentals with Python Scapy | by Moraneus - Medium, accessed September 7, 2025, [https://medium.com/@moraneus/exploring-network-fundamentals-with-python-scapy-c21024813285](https://medium.com/@moraneus/exploring-network-fundamentals-with-python-scapy-c21024813285)  
24. Network Scanning using scapy module - Python - GeeksforGeeks, accessed September 7, 2025, [https://www.geeksforgeeks.org/python/network-scanning-using-scapy-module-python/](https://www.geeksforgeeks.org/python/network-scanning-using-scapy-module-python/)  
25. Scapy Sniffing with Custom Actions, Part 1 - thePacketGeek, accessed September 7, 2025, [https://thepacketgeek.com/scapy/sniffing-custom-actions/part-1/](https://thepacketgeek.com/scapy/sniffing-custom-actions/part-1/)  
26. How to Sniff HTTP Packets in the Network using Scapy in Python, accessed September 7, 2025, [https://thepythoncode.com/article/sniff-http-packets-scapy-python](https://thepythoncode.com/article/sniff-http-packets-scapy-python)  
27. Packet Sniffing: Scapy Basics - Tutorial | Krython, accessed September 7, 2025, [https://www.krython.com/tutorial/python/packet-sniffing-scapy-basics](https://www.krython.com/tutorial/python/packet-sniffing-scapy-basics)  
28. Adding new protocols — Scapy 2.6.1 documentation, accessed September 7, 2025, [https://scapy.readthedocs.io/en/latest/build_dissect.html](https://scapy.readthedocs.io/en/latest/build_dissect.html)  
29. Adding custom pmu layer to scapy - P4 Programming Language, accessed September 7, 2025, [https://forum.p4.org/t/adding-custom-pmu-layer-to-scapy/660](https://forum.p4.org/t/adding-custom-pmu-layer-to-scapy/660)
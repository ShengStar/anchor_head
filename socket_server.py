import socket
import numpy as np
import pickle
import struct

hostname = '172.20.107.96'
port = 55555
addr = (hostname, port)
srv = socket.socket()
srv.bind(addr)
srv.listen(5)
print("Server waitting for connection")
# connect_socket, client_addr = srv.accept()
# print("Connected client: ", client_addr)
post_i = 0

connect_socket, client_addr = srv.accept()
print("Connected client: ", client_addr)
rec_size = connect_socket.recv(1024)
size = struct.unpack('i', rec_size)[0]
connect_socket.send(bytes("send data", encoding='utf-8'))


rec_data = b''
while (len(rec_data)) < size:
    rec_data += connect_socket.recv(1024)
np_array = pickle.loads(rec_data)
np_array = np_array.reshape(-1,4)
print(np_array.size)
# np.save('data/000000.npy',np_array)
connect_socket.send(bytes("ok", encoding='utf-8'))
connect_socket.close()

while True:

    bin_idex = '%06d' % (post_i + 1)

    connect_socket, client_addr = srv.accept()
    print("Connected client: ", client_addr)
    rec_size = connect_socket.recv(1024)
    print("size")
    size = struct.unpack('i', rec_size)[0]
    connect_socket.send(bytes("send data", encoding='utf-8'))
    
    
    rec_data = b''
    while (len(rec_data)) < size:
        rec_data += connect_socket.recv(1024)
    np_array = pickle.loads(rec_data)
    np_array = np_array.reshape(-1,4)
    print(np_array.size)
    # np.save('data/'+str(bin_idex)+'.npy',np_array)
    connect_socket.send(bytes("ok", encoding='utf-8'))
    connect_socket.close()
    print("input close")

    connect_socket, client_addr = srv.accept()
    print("Connected client: ", client_addr)

    image_idex = '%06d' % post_i
    rec_data = b''
    while True:
        rec = connect_socket.recv(1024)
        rec_data += rec
        if(len(rec_data)==65702):
            break
    np_array1 = pickle.loads(rec_data)
    np_array1 = np_array1.reshape(1,64,128,2)
    print(np_array1.size)
    # np.save('result/'+str(image_idex)+'_cls_preds.npy',np_array1)
    connect_socket.send(bytes("ok", encoding='utf-8'))
    
    rec_data = b''
    while True:
        rec = connect_socket.recv(1024)
        rec_data += rec
        if(len(rec_data)==458918):
            break      
    np_array2 = pickle.loads(rec_data)
    np_array2 = np_array2.reshape(1,64,128,14)
    print(np_array2.size)
    # np.save('result/'+str(image_idex)+'_box_preds.npy',np_array2)
    connect_socket.send(bytes("ok", encoding='utf-8'))
    
    rec_data = b''
    while True:
        rec = connect_socket.recv(1024)
        rec_data += rec
        if(len(rec_data)==131238):
            break      
    np_array3 = pickle.loads(rec_data)
    np_array3 = np_array3.reshape(1,64,128,4)
    print(np_array3.size)
    # np.save('result/'+str(image_idex)+'_dir_cls_preds.npy',np_array3)
    connect_socket.send(bytes("ok", encoding='utf-8'))
    connect_socket.close()
    
    post_i += 1

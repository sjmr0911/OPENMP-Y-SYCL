// suma_sycl_adaptado.cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define N 1048576
#define THREADS_PER_BLOCK 256

using namespace sycl;

int main() {
    std::vector<float> h_A(N), h_B(N), h_C(N);

    // Inicialización de vectores
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 2.0f;
    }

    // Seleccionar dispositivo y crear cola con perfil activo para medir tiempo
    queue q(cpu_selector_v, property::queue::enable_profiling{});
    std::cout << "Dispositivo usado: " << q.get_device().get_info<info::device::name>() << "\n";

    // Crear buffers para A, B, C (manejan transferencia automáticamente)
    buffer<float> A_buf(h_A.data(), range<1>(N));
    buffer<float> B_buf(h_B.data(), range<1>(N));
    buffer<float> C_buf(h_C.data(), range<1>(N));

    // Definir el rango de ejecución similar a CUDA (nd_range)
    range<1> global_range((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK);
    range<1> local_range(THREADS_PER_BLOCK);

    // Lanzar kernel y obtener evento para medir tiempo
    auto event = q.submit([&](handler& h) {
        auto a = A_buf.get_access<access::mode::read>(h);
        auto b = B_buf.get_access<access::mode::read>(h);
        auto c = C_buf.get_access<access::mode::write>(h);

        // nd_range define tamaño global y local (como bloques/hilos CUDA)
        h.parallel_for(nd_range<1>(global_range, local_range), [=](nd_item<1> item) {
            size_t i = item.get_global_id(0);
            if (i < N) {
                c[i] = a[i] + b[i];
            }
        });
    });

    event.wait();

    // Medir tiempo en segundos con perfil del evento
    auto start_time = event.get_profiling_info<info::event_profiling::command_start>();
    auto end_time = event.get_profiling_info<info::event_profiling::command_end>();
    double elapsed = (end_time - start_time) * 1e-9;

    std::cout << "Tiempo kernel (GPU) SYCL: " << elapsed << " segundos\n";

    // Acceso host para verificar resultados (usar host_accessor para sincronizar)
    host_accessor h_c(C_buf, read_only);

    // Verificación
    int errores = 0;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(h_c[i] - (h_A[i] + h_B[i])) > 1e-5) {
            errores++;
            std::cout << "Error en índice " << i << ": " << h_c[i] << " != " << h_A[i] + h_B[i] << "\n";
            break;
        }
    }

    std::cout << "Verificación: " << (errores ? "FALLÓ" : "ÉXITO") << std::endl;

    return 0;
}




# %%

import asyncio
import time

# %%

async def tiempo(tarea):
    print(f'tarea {tarea} tardo {time.time-start_time:.5f} segundos')

#cocer pasta
async def cocer_pasta():
    print('Iniciamos tarea: cocer pasta')
    await asyncio.sleep(1)
    print('poner agua a hervir')
    await asyncio.sleep(4)
    print('agua caliente')
    await asyncio.sleep(1)
    print('meter pasta')
    await asyncio.sleep(8)
    print('cocer pasta lista')
#preparar_salsa
async def preparar_salsa():
    print('Iniciamos tarea: preparar salsa')
    await asyncio.sleep(1)
    print('corta ingredientes')
    await asyncio.sleep(2)
    print('poner carne')
    await asyncio.sleep(3)
    print('verduras')
    await asyncio.sleep(4)
    print('tomate')
    await asyncio.sleep(2)
    print('preparar salsa hecha')

async def main():
    await asyncio.gather(cocer_pasta(), preparar_salsa())
    # orquestamos

if __name__ == '__main__':
    start_time = time.time()
    asyncio.run(main())
    asyncio.run(main())
    end_time = time.time()
    print(f'tarea terminada en {end_time - start_time}')
# emplatar
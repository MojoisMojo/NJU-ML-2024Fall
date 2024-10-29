from model import main
import logging
import time
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    time_stemp = time.strftime("%m%d_%H%M%S", time.localtime())
    btz = 16
    for e,lr in [(1000,0.01),(200,0.01),(100,0.1),(50,1)]:
        for inmethod in ["random","xavier","he"]:
            print("#"*50,f"\ne={e},lr={lr},btz={btz},inmethod={inmethod}")
            nn = main(time_stemp=time_stemp, e=e, lr=lr, btz=btz, inmethod=inmethod)
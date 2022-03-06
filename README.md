### frangifilter


```
 weekend homework to refresh my memory on the classic frangi filter.
```



+ build and go into container.

```
bash build.sh

docker run -it -u $(id -u):$(id -g) -w /workdir -v $PWD:/workdir frangifilter bash
```


+ download and segment image using classical image processing methods.

```
python demo.py
```



### output images

lung mask:

![alt text](static/mip_lung.png)

vessel enhanced image:

![alt text](static/mip_vessel.png)

fissure enhanced image  (pending param tweaking)

![alt text](static/mip_fissure.png)

airway enhanced image (pending param tweaking)

![alt text](static/mip_airway.png)


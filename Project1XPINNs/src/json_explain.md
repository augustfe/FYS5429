The JSON format for the domain decomposition should look as follows
```JSON
{
    "XPINNs":[
        {
            "Internal points":[[0.5, 0.5], [0.7, 0.6], [0.8,0.9]],
            "Boundary points":[[0,0], [1,0], [0.5,0]]
        },
        {
            "Internal points":[[0.5, 1.5], [0.7, 1.6], [0.8,1.9]],
            "Boundary points":[[1,2], [1.5,2], [0.5,2]]
        }
    ],
    "Interfaces": [
        {
            "XPINNs":[0,1],  # XPINNs are denoted by their index in the original XPINNs array
            "Points":[[1,1],[0.2,1],[1.5,1]] # These are the shared points
        }
    ]
}
```
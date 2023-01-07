using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class gameOver : MonoBehaviour
{
    public void Setup()
    {
        // make text visible
        gameObject.SetActive(true);
    }

    public void hide()
    {
        // make text invisible
        gameObject.SetActive(false);
    }
}

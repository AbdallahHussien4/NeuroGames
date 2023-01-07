using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class healthBar : MonoBehaviour
{
    public Slider slider;
    public Gradient gradiant;
    public Image fill;
    
    // public function used to set health value and set the bar color
    public void setHealth(int health)
    {
        slider.value = health;
        fill.color = gradiant.Evaluate(slider.normalizedValue);
    }


    // set the maximum value of health
    public void setMaxHealth(int health)
    {
        slider.maxValue = health;
        slider.value = health;
        fill.color = gradiant.Evaluate(1f);
    }


    // update health
    public void takeDamage(int num)
    {
        if (slider.value != 0)
        {
            slider.value -= num;
            fill.color = gradiant.Evaluate(slider.normalizedValue);
        }
    }

    // return health value
    public float getValue()
    {
        return slider.value;
    }

    
    void Start()
    {
        setMaxHealth(100);
    }
}

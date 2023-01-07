using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class leftHitScript : StateMachineBehaviour
{
    public AudioClip hit;
    AudioSource source;

    // OnStateEnter is called when a transition starts and the state machine starts to evaluate this state
    override public void OnStateEnter(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        
        // play sound effect once entered hit state
        source = GameObject.FindWithTag("fighter").GetComponent<fighter>().GetComponent<AudioSource>();
        source.PlayOneShot(hit);

    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class fighter : MonoBehaviour
{
    Animator m_Animator;
    AudioSource source;
    bool newRound = true;
    
    public Transform vrPlayer;
    public AudioClip bell;
    public Animator leftBox;
    public Animator rightBox;
    public healthBar enemy;
    public healthBar player;
    public win win_screen;
    public gameOver over_screen;

    // Start is called before the first frame update
    void Start()
    {
        // getting variables from game objects
        m_Animator = gameObject.GetComponent<Animator>();
        source = gameObject.GetComponent<AudioSource>();  
        
        // play sound of bell
        source.PlayOneShot(bell);
        m_Animator.SetBool("key", false);

    }
    IEnumerator HitEffect()
    {
        // Being here means Idle state finished
        yield return new WaitForSecondsRealtime(5.0f);

        // Generate random number for attack 
        float num = Random.Range(0.0f,1.0f);
        
        // Perform the action
        if (num < 0.3 && newRound == true)
        {
            // check if player is in idle state and fighter not performing an action
            if ( leftBox.GetCurrentAnimatorStateInfo(0).IsName("idle") && rightBox.GetCurrentAnimatorStateInfo(0).IsName("idle"))
            {
                m_Animator.SetTrigger("attack1");
            }
        }
        else if ( num < 0.6 && newRound == true)
        {
            // check if player is in idle state and fighter not performing an action
            if ( leftBox.GetCurrentAnimatorStateInfo(0).IsName("idle") && rightBox.GetCurrentAnimatorStateInfo(0).IsName("idle"))
            {
                m_Animator.SetTrigger("attack2");
            }
        }

        m_Animator.SetBool("key", false);
        
    }

    // Start New Round
    IEnumerator Restart()
    {
        yield return new WaitForSecondsRealtime(7.0f);

        // resetting parameters and scene
        newRound = true;
        enemy.setHealth(100);
        player.setHealth(100);
        win_screen.hide();
        over_screen.hide();
        source.PlayOneShot(bell);
        m_Animator.SetBool("key", false);
    }

    // Update is called once per frame
    void Update()
    {

        if (m_Animator.GetCurrentAnimatorStateInfo(0).IsName("idle") && m_Animator.GetBool("key") == false)
        {
            // check both players health 
            if(enemy.getValue() > 0  && player.getValue() > 0)
            {
                m_Animator.SetBool("key",true);
                StartCoroutine(HitEffect());
            }
        }  


        // player won
        if(enemy.getValue() <= 0  && player.getValue() > 0 && newRound == true)
        {
            newRound = false;
            win_screen.Setup();
            StartCoroutine(Restart());

            
        // fighter won
        } else if(enemy.getValue() > 0  && player.getValue() <= 0 && newRound == true)
        {
            newRound = false;
            over_screen.Setup();
            StartCoroutine(Restart());
        }

    }
}
